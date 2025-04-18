// deepcomplete/deepcomplete.go
package deepcomplete

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256" // For hashing
	"encoding/gob"  // For serializing cache data
	"encoding/hex"
	"encoding/json"
	"errors" // Using errors.Join requires Go 1.20+
	"fmt"
	"go/ast"
	"go/format" // For formatting receiver in preamble
	"go/parser"
	"go/token"
	"go/types"
	"io" // For file hashing errors
	"log"
	"net" // For network errors
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug" // For panic recovery stack trace
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8" // Added for LSP position conversion helpers

	"go.etcd.io/bbolt"                  // Import bbolt
	"golang.org/x/tools/go/ast/astutil" // Utility for AST traversal
	"golang.org/x/tools/go/packages"    // Standard way to load packages for analysis
	// Removed: "golang.org/x/tools/go/types/exportdata" - No longer used due to instability/moves
)

// =============================================================================
// Constants & Core Types
// =============================================================================

const (
	defaultOllamaURL = "http://localhost:11434"
	defaultModel     = "deepseek-coder-r2"
	// Standard prompt template
	promptTemplate = `<s>[INST] <<SYS>>
You are an expert Go programming assistant.
Analyze the provided context (enclosing function/method, imports, scope variables, comments, code structure) and the preceding code snippet.
Complete the Go code accurately and concisely.
Output ONLY the raw Go code completion, without any markdown, explanations, or introductory text.
<</SYS>>

CONTEXT:
%s

CODE SNIPPET TO COMPLETE:
` + "```go\n%s\n```" + `
[/INST]`
	// FIM prompt template
	fimPromptTemplate = `<s>[INST] <<SYS>>
You are an expert Go programming assistant performing a fill-in-the-middle task.
Analyze the provided context and the code surrounding the <MID> marker.
Insert Go code at the <MID> marker to logically connect the <PRE>fix and <SUF>fix code blocks.
Output ONLY the raw Go code completion for the middle part, without any markdown, explanations, or introductory text.
<</SYS>>

CONTEXT:
%s

CODE TO FILL:
<PRE>%s<MID>%s<SUF>
[/INST]`

	maxRetries       = 3
	retryDelay       = 2 * time.Second
	defaultMaxTokens = 256
	// DefaultStop is exported for potential use in CLI default flags
	DefaultStop        = "\n"
	defaultTemperature = 0.1
	// Default config file name
	defaultConfigFileName = "config.json"
	configDirName         = "deepcomplete" // Subdirectory name for config/data

	// --- Cache Constants ---
	// Increment cache schema version due to change in cached data structure
	cacheSchemaVersion = 2
)

var (
	// --- Cache Bucket Name ---
	cacheBucketName = []byte("AnalysisCache")
)

// Config holds the active configuration for the autocompletion. Exported.
type Config struct {
	OllamaURL      string   `json:"ollama_url"`
	Model          string   `json:"model"`
	PromptTemplate string   `json:"-"` // Loaded internally, not from file
	FimTemplate    string   `json:"-"` // Loaded internally, not from file
	MaxTokens      int      `json:"max_tokens"`
	Stop           []string `json:"stop"`
	Temperature    float64  `json:"temperature"`
	UseAst         bool     `json:"use_ast"` // Use AST and Type analysis (preferred)
	UseFim         bool     `json:"use_fim"` // Use Fill-in-the-Middle prompting
	MaxPreambleLen int      `json:"max_preamble_len"`
	MaxSnippetLen  int      `json:"max_snippet_len"`
}

// Validate checks if the configuration values are valid.
func (c *Config) Validate() error {
	if strings.TrimSpace(c.OllamaURL) == "" {
		return errors.New("ollama_url cannot be empty")
	}
	if _, err := url.ParseRequestURI(c.OllamaURL); err != nil {
		return fmt.Errorf("invalid ollama_url: %w", err)
	}
	if strings.TrimSpace(c.Model) == "" {
		return errors.New("model cannot be empty")
	}
	if c.MaxTokens <= 0 {
		log.Printf("Warning: max_tokens (%d) is not positive, using default %d",
			c.MaxTokens, defaultMaxTokens)
		c.MaxTokens = defaultMaxTokens
	}
	if c.Temperature < 0 {
		log.Printf("Warning: temperature (%.2f) is negative, using default %.2f",
			c.Temperature, defaultTemperature)
		c.Temperature = defaultTemperature
	}
	if c.MaxPreambleLen <= 0 {
		log.Printf("Warning: max_preamble_len (%d) is not positive, using default %d",
			c.MaxPreambleLen, DefaultConfig.MaxPreambleLen)
		c.MaxPreambleLen = DefaultConfig.MaxPreambleLen
	}
	if c.MaxSnippetLen <= 0 {
		log.Printf("Warning: max_snippet_len (%d) is not positive, using default %d",
			c.MaxSnippetLen, DefaultConfig.MaxSnippetLen)
		c.MaxSnippetLen = DefaultConfig.MaxSnippetLen
	}
	return nil
}

// FileConfig represents the structure of the JSON config file.
type FileConfig struct {
	OllamaURL      *string   `json:"ollama_url"`
	Model          *string   `json:"model"`
	MaxTokens      *int      `json:"max_tokens"`
	Stop           *[]string `json:"stop"`
	Temperature    *float64  `json:"temperature"`
	UseAst         *bool     `json:"use_ast"`
	UseFim         *bool     `json:"use_fim"`
	MaxPreambleLen *int      `json:"max_preamble_len"`
	MaxSnippetLen  *int      `json:"max_snippet_len"`
}

// AstContextInfo holds structured information extracted from AST/Types analysis. Exported.
// This struct is populated during analysis and used to build the prompt preamble.
// It is NOT directly cached anymore.
type AstContextInfo struct {
	FilePath           string
	CursorPos          token.Pos
	PackageName        string
	EnclosingFunc      *types.Func   // Function or method type object
	EnclosingFuncNode  *ast.FuncDecl // Function or method AST node
	ReceiverType       string        // Formatted receiver type if it's a method
	EnclosingBlock     *ast.BlockStmt
	Imports            []*ast.ImportSpec
	CommentsNearCursor []string
	IdentifierAtCursor *ast.Ident
	IdentifierType     types.Type
	IdentifierObject   types.Object
	SelectorExpr       *ast.SelectorExpr
	SelectorExprType   types.Type
	CallExpr           *ast.CallExpr
	CallExprFuncType   *types.Signature
	CallArgIndex       int
	ExpectedArgType    types.Type
	CompositeLit       *ast.CompositeLit
	CompositeLitType   types.Type
	VariablesInScope   map[string]types.Object
	PromptPreamble     string // Built during analysis OR retrieved from cache
	AnalysisErrors     []error
}

// SnippetContext holds the code prefix and suffix for prompting.
type SnippetContext struct {
	Prefix   string
	Suffix   string
	FullLine string // The full line where the cursor is located
}

// OllamaError defines a custom error for Ollama API issues.
type OllamaError struct {
	Message string
	Status  int
}

// Error implements the error interface for OllamaError.
func (e *OllamaError) Error() string {
	return fmt.Sprintf("Ollama error: %s (Status: %d)", e.Message, e.Status)
}

// OllamaResponse represents the streaming response structure.
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"`
}

// --- Cache Data Structures (Exportdata Fix) ---

// CachedAnalysisData holds the derived information extracted from analysis
// that is needed to reconstruct the prompt preamble. This struct is gob-encoded
// and stored in the cache.
type CachedAnalysisData struct {
	PackageName    string
	PromptPreamble string
	// Add other simple fields derived from AstContextInfo if needed directly
	// from cache without re-analysis.
}

// CachedAnalysisEntry represents the structure stored in the bbolt database.
// It includes metadata (hashes) and the gob-encoded analysis results.
type CachedAnalysisEntry struct {
	SchemaVersion   int
	GoModHash       string
	InputFileHashes map[string]string // Key: relative path from package dir, Value: SHA256 hash
	AnalysisGob     []byte            // Gob-encoded CachedAnalysisData
}

// LSPPosition represents the 0-based line and character offset used by the Language Server Protocol.
type LSPPosition struct {
	Line      uint32 // 0-based line number
	Character uint32 // 0-based UTF-16 code unit offset from the start of the line
}

// MemberKind defines the type of member (field or method). Restored.
type MemberKind string

const (
	FieldMember  MemberKind = "field"
	MethodMember MemberKind = "method"
	OtherMember  MemberKind = "other"
)

// MemberInfo holds structured information about a type member. Restored.
type MemberInfo struct {
	Name       string
	Kind       MemberKind
	TypeString string
}

// =============================================================================
// Exported Errors
// =============================================================================

var (
	ErrAnalysisFailed     = errors.New("code analysis failed")
	ErrOllamaUnavailable  = errors.New("ollama API unavailable or returned server error")
	ErrStreamProcessing   = errors.New("error processing LLM stream")
	ErrConfig             = errors.New("configuration error")
	ErrInvalidConfig      = errors.New("invalid configuration")
	ErrCache              = errors.New("cache operation failed")
	ErrPositionConversion = errors.New("position conversion failed")
)

// =============================================================================
// Interfaces for Components
// =============================================================================

// LLMClient defines the interface for interacting with the LLM API.
type LLMClient interface {
	GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error)
}

// Analyzer defines the interface for analyzing code context. Exported for testability.
type Analyzer interface {
	Analyze(ctx context.Context, filename string, line, col int) (*AstContextInfo, error)
	Close() error
	InvalidateCache(dir string) error
}

// PromptFormatter defines the interface for formatting the final prompt.
type PromptFormatter interface {
	FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string
}

// =============================================================================
// Variables & Default Config
// =============================================================================

var (
	// DefaultConfig enables AST context by default. Exported.
	DefaultConfig = Config{
		OllamaURL:      defaultOllamaURL,
		Model:          defaultModel,
		PromptTemplate: promptTemplate,
		FimTemplate:    fimPromptTemplate,
		MaxTokens:      defaultMaxTokens,
		Stop:           []string{DefaultStop, "}", "//", "/*"},
		Temperature:    defaultTemperature,
		UseAst:         true,
		UseFim:         false,
		MaxPreambleLen: 2048,
		MaxSnippetLen:  2048,
	}

	// Pastel color codes for terminal output. Exported.
	ColorReset  = "\033[0m"
	ColorGreen  = "\033[38;5;119m"
	ColorYellow = "\033[38;5;220m"
	ColorBlue   = "\033[38;5;153m"
	ColorRed    = "\033[38;5;203m"
	ColorCyan   = "\033[38;5;141m"
)

// =============================================================================
// Exported Helper Functions (e.g., for CLI)
// =============================================================================

// PrettyPrint prints colored text to the terminal. Exported for CLI use.
func PrettyPrint(color, text string) {
	fmt.Print(color, text, ColorReset)
}

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadConfig loads configuration from standard locations, merges with defaults,
// and attempts to write a default config file if none exists.
func LoadConfig() (Config, error) {
	cfg := DefaultConfig
	var loadedFromFile bool
	var loadErrors []error
	var configParseError error

	primaryPath, secondaryPath, pathErr := getConfigPaths()
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		log.Printf("Warning: Could not determine config paths: %v", pathErr)
	}

	// Try loading from primary path first
	if primaryPath != "" {
		loaded, loadErr := loadAndMergeConfig(primaryPath, &cfg)
		if loadErr != nil {
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", primaryPath, loadErr))
		}
		loadedFromFile = loaded
		if loadedFromFile && loadErr == nil {
			log.Printf("Loaded config from %s", primaryPath)
		} else if loadedFromFile && loadErr != nil {
			log.Printf("Attempted load from %s but failed.", primaryPath)
		}
	}

	// Try secondary path ONLY if primary was not found OR if primary existed but failed reading/parsing
	primaryNotFound := len(loadErrors) > 0 && errors.Is(loadErrors[len(loadErrors)-1], os.ErrNotExist)

	if !loadedFromFile && secondaryPath != "" && (primaryNotFound || primaryPath == "") {
		loaded, loadErr := loadAndMergeConfig(secondaryPath, &cfg)
		if loadErr != nil {
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				if configParseError == nil {
					configParseError = loadErr
				}
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", secondaryPath, loadErr))
		}
		loadedFromFile = loaded
		if loadedFromFile && loadErr == nil {
			log.Printf("Loaded config from %s", secondaryPath)
		} else if loadedFromFile && loadErr != nil {
			log.Printf("Attempted load from %s but failed.", secondaryPath)
		}
	}

	// Write default config if no file existed OR if parsing failed
	loadSucceeded := loadedFromFile && configParseError == nil
	if !loadSucceeded {
		if configParseError != nil {
			log.Printf("Existing config file failed to parse: %v. Attempting to write default.", configParseError)
		} else {
			log.Println("No config file found. Attempting to write default.")
		}
		writePath := primaryPath
		if writePath == "" {
			writePath = secondaryPath
		}
		if writePath != "" {
			log.Printf("Attempting to write default config to %s", writePath)
			if err := writeDefaultConfig(writePath, DefaultConfig); err != nil {
				log.Printf("Warning: Failed to write default config: %v", err)
				loadErrors = append(loadErrors, fmt.Errorf("writing default config failed: %w", err))
			}
		} else {
			log.Println("Warning: Cannot determine path to write default config.")
			loadErrors = append(loadErrors, errors.New("cannot determine default config path"))
		}
		cfg = DefaultConfig // Use defaults if load failed
	}

	// Assign internal templates
	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	// Validate final config
	finalCfg := cfg
	if err := finalCfg.Validate(); err != nil {
		log.Printf("Warning: Config after load/merge failed validation: %v. Returning defaults.", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		if valErr := DefaultConfig.Validate(); valErr != nil {
			return DefaultConfig, fmt.Errorf("default config is invalid: %w", valErr)
		}
		finalCfg = DefaultConfig
	}

	// Return validated config and any non-fatal load errors
	if len(loadErrors) > 0 {
		return finalCfg, fmt.Errorf("%w: %w", ErrConfig, errors.Join(loadErrors...))
	}
	return finalCfg, nil
}

// getConfigPaths determines the primary (XDG) and secondary (~/.local/share) config paths.
func getConfigPaths() (primary string, secondary string, err error) {
	var cfgErr, homeErr error
	userConfigDir, cfgErr := os.UserConfigDir()
	if cfgErr == nil {
		primary = filepath.Join(userConfigDir, configDirName, defaultConfigFileName)
	} else {
		log.Printf("Warning: Could not determine user config directory: %v", cfgErr)
	}
	homeDir, homeErr := os.UserHomeDir()
	if homeErr == nil {
		if primary == "" && cfgErr != nil { // Fallback path using HOME if XDG fails
			primary = filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
			log.Printf("Using fallback primary config path: %s", primary)
		}
		secondary = filepath.Join(homeDir, ".local", "share", configDirName, defaultConfigFileName)
	} else {
		log.Printf("Warning: Could not determine user home directory: %v", homeErr)
	}
	if primary == "" && secondary == "" { // Report error only if BOTH failed
		err = fmt.Errorf("cannot determine config/home directories: config error: %v; home error: %v", cfgErr, homeErr)
	}
	return primary, secondary, err
}

// loadAndMergeConfig attempts to load config from a path and merge it into cfg.
func loadAndMergeConfig(path string, cfg *Config) (bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return true, fmt.Errorf("reading config file %q failed: %w", path, err)
	}
	if len(data) == 0 {
		log.Printf("Warning: Config file exists but is empty: %s", path)
		return true, nil
	}

	var fileCfg FileConfig
	if err := json.Unmarshal(data, &fileCfg); err != nil {
		return true, fmt.Errorf("parsing config file JSON %q failed: %w", path, err)
	}

	// Merge fields if they exist in the file config
	if fileCfg.OllamaURL != nil {
		cfg.OllamaURL = *fileCfg.OllamaURL
	}
	if fileCfg.Model != nil {
		cfg.Model = *fileCfg.Model
	}
	if fileCfg.MaxTokens != nil {
		cfg.MaxTokens = *fileCfg.MaxTokens
	}
	if fileCfg.Stop != nil {
		cfg.Stop = *fileCfg.Stop
	}
	if fileCfg.Temperature != nil {
		cfg.Temperature = *fileCfg.Temperature
	}
	if fileCfg.UseAst != nil {
		cfg.UseAst = *fileCfg.UseAst
	}
	if fileCfg.UseFim != nil {
		cfg.UseFim = *fileCfg.UseFim
	}
	if fileCfg.MaxPreambleLen != nil {
		cfg.MaxPreambleLen = *fileCfg.MaxPreambleLen
	}
	if fileCfg.MaxSnippetLen != nil {
		cfg.MaxSnippetLen = *fileCfg.MaxSnippetLen
	}

	return true, nil
}

// writeDefaultConfig creates the directory and writes the default config as JSON.
func writeDefaultConfig(path string, defaultConfig Config) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0750); err != nil {
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}
	type ExportableConfig struct {
		OllamaURL      string   `json:"ollama_url"`
		Model          string   `json:"model"`
		MaxTokens      int      `json:"max_tokens"`
		Stop           []string `json:"stop"`
		Temperature    float64  `json:"temperature"`
		UseAst         bool     `json:"use_ast"`
		UseFim         bool     `json:"use_fim"`
		MaxPreambleLen int      `json:"max_preamble_len"`
		MaxSnippetLen  int      `json:"max_snippet_len"`
	}
	expCfg := ExportableConfig{
		OllamaURL:      defaultConfig.OllamaURL,
		Model:          defaultConfig.Model,
		MaxTokens:      defaultConfig.MaxTokens,
		Stop:           defaultConfig.Stop,
		Temperature:    defaultConfig.Temperature,
		UseAst:         defaultConfig.UseAst,
		UseFim:         defaultConfig.UseFim,
		MaxPreambleLen: defaultConfig.MaxPreambleLen,
		MaxSnippetLen:  defaultConfig.MaxSnippetLen,
	}
	jsonData, err := json.MarshalIndent(expCfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal default config to JSON: %w", err)
	}
	if err := os.WriteFile(path, jsonData, 0640); err != nil {
		return fmt.Errorf("failed to write default config file %s: %w", path, err)
	}
	log.Printf("Wrote default configuration to %s", path)
	return nil
}

// =============================================================================
// Default Implementations (Continued)
// =============================================================================

// --- Ollama Client ---

// httpOllamaClient implements LLMClient using standard HTTP calls.
type httpOllamaClient struct {
	httpClient *http.Client
}

// newHttpOllamaClient creates a new client for Ollama interaction. (unexported)
func newHttpOllamaClient() *httpOllamaClient {
	return &httpOllamaClient{httpClient: &http.Client{Timeout: 90 * time.Second}}
}

// GenerateStream handles the HTTP request to the Ollama generate endpoint.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	base := strings.TrimSuffix(config.OllamaURL, "/")
	endpointPath := "/api/generate"
	endpointURL := base + endpointPath

	u, err := url.Parse(endpointURL)
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL '%s': %w", endpointURL, err)
	}

	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"temperature": config.Temperature,
			"num_ctx":     4096,
			"top_p":       0.9,
			"stop":        config.Stop,
			"num_predict": config.MaxTokens,
		},
	}
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/x-ndjson")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		var netErr *net.OpError
		if errors.As(err, &netErr) && netErr.Op == "dial" {
			return nil, fmt.Errorf("%w: connection refused or network error connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		return nil, fmt.Errorf("%w: error making HTTP request: %w", ErrOllamaUnavailable, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(read error)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error
			}
		}
		err = &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
	}
	return resp.Body, nil
}

// --- Code Analyzer (bbolt implementation) ---

// GoPackagesAnalyzer implements Analyzer using go/packages and bbolt cache.
type GoPackagesAnalyzer struct {
	db *bbolt.DB
	mu sync.Mutex
}

// NewGoPackagesAnalyzer creates a new Go code analyzer, opening the bbolt DB.
func NewGoPackagesAnalyzer() *GoPackagesAnalyzer {
	dbPath := ""
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil {
			dbPath = filepath.Join(dbDir, "analysis_cache.db")
		} else {
			log.Printf("Warning: Could not create bbolt cache directory %s: %v", dbDir, err)
		}
	} else {
		log.Printf("Warning: Could not determine user cache directory: %v. Caching disabled.", err)
	}

	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second}
		db, err = bbolt.Open(dbPath, 0600, opts)
		if err != nil {
			log.Printf("Warning: Failed to open bbolt cache file %s: %v. Caching will be disabled.", dbPath, err)
			db = nil
		} else {
			err = db.Update(func(tx *bbolt.Tx) error {
				_, err := tx.CreateBucketIfNotExists(cacheBucketName)
				if err != nil {
					return fmt.Errorf("failed to create cache bucket %s: %w", string(cacheBucketName), err)
				}
				return nil
			})
			if err != nil {
				log.Printf("Warning: Failed to ensure bbolt bucket exists: %v. Caching disabled.", err)
				db.Close()
				db = nil
			} else {
				log.Printf("Using bbolt cache at %s (Schema v%d)", dbPath, cacheSchemaVersion)
			}
		}
	}
	return &GoPackagesAnalyzer{db: db}
}

// Close closes the underlying bbolt database.
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.db != nil {
		log.Println("Closing bbolt cache database.")
		err := a.db.Close()
		a.db = nil
		return err
	}
	return nil
}

// Analyze parses the file, performs type checking (potentially using cache for preamble), and extracts context.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, filename string, line, col int) (info *AstContextInfo, analysisErr error) {
	info = &AstContextInfo{
		FilePath:         filename,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		CallArgIndex:     -1,
	}
	defer func() {
		r := recover()
		if r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			addAnalysisError(info, panicErr)
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr)
			}
			log.Printf("Panic recovered during AnalyzeCodeContext: %v\n%s", r, string(debug.Stack()))
		}
	}()

	absFilename, err := filepath.Abs(filename)
	if err != nil {
		return info, fmt.Errorf("failed to get absolute path for '%s': %w", filename, err)
	}
	info.FilePath = absFilename
	log.Printf("Starting context analysis for: %s (%d:%d)", absFilename, line, col)
	dir := filepath.Dir(absFilename)

	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	var loadDuration, stepsDuration, preambleDuration time.Duration
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry

	// --- Try Loading from Cache ---
	if a.db != nil {
		readStart := time.Now()
		dbErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				return fmt.Errorf("%w: cache bucket %s not found during View", ErrCache, string(cacheBucketName))
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				return nil
			} // Cache miss

			var decoded CachedAnalysisEntry
			decoder := gob.NewDecoder(bytes.NewReader(valBytes))
			if err := decoder.Decode(&decoded); err != nil {
				log.Printf("Warning: Failed to gob-decode cached entry header for key %s: %v. Treating as miss.", string(cacheKey), err)
				return nil
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				log.Printf("Warning: Cache data for key %s has old schema version %d (want %d). Ignoring.", string(cacheKey), decoded.SchemaVersion, cacheSchemaVersion)
				return nil
			}
			cachedEntry = &decoded
			return nil
		})
		if dbErr != nil {
			log.Printf("Warning: Error reading from bbolt cache: %v", dbErr)
			addAnalysisError(info, fmt.Errorf("%w: read error: %w", ErrCache, dbErr))
		}
		log.Printf("DEBUG: Cache read attempt took %v", time.Since(readStart))
	} else {
		log.Printf("DEBUG: Cache disabled (db handle is nil).")
	}

	// --- Validate Cache Hit and Use Cached Data ---
	if cachedEntry != nil {
		validationStart := time.Now()
		log.Printf("DEBUG: Potential cache hit for key: %s. Validating file hashes...", string(cacheKey))
		currentHashes, hashErr := calculateInputHashes(dir, nil)
		if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes) {
			log.Printf("DEBUG: Cache VALID for key: %s. Attempting to decode analysis data...", string(cacheKey))
			decodeStart := time.Now()
			var analysisData CachedAnalysisData
			decoder := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob))
			if decodeErr := decoder.Decode(&analysisData); decodeErr == nil {
				// Cache hit: Populate info directly from cached data
				info.PackageName = analysisData.PackageName
				info.PromptPreamble = analysisData.PromptPreamble
				cacheHit = true
				loadDuration = time.Since(decodeStart) // Record decode time
				log.Printf("DEBUG: Analysis data successfully decoded from cache in %v.", loadDuration)
				log.Printf("DEBUG: Using cached preamble (length %d). Skipping packages.Load and analysis steps.", len(info.PromptPreamble))
			} else {
				log.Printf("Warning: Failed to gob-decode cached analysis data: %v. Treating as miss.", decodeErr)
				addAnalysisError(info, fmt.Errorf("%w: analysis data decode error: %w", ErrCache, decodeErr))
				a.deleteCacheEntry(cacheKey)
			}
		} else {
			log.Printf("DEBUG: Cache INVALID for key: %s (HashErr: %v, Hash Match: %t, GoMod Match: %t). Treating as miss.",
				string(cacheKey), hashErr, compareFileHashes(currentHashes, cachedEntry.InputFileHashes), cachedEntry.GoModHash == goModHash)
			a.deleteCacheEntry(cacheKey)
			if hashErr != nil {
				addAnalysisError(info, fmt.Errorf("%w: hash calculation error: %w", ErrCache, hashErr))
			}
		}
		log.Printf("DEBUG: Cache validation/decode took %v", time.Since(validationStart))
	}

	// --- Perform Full Analysis if Cache Miss ---
	if !cacheHit {
		if a.db == nil {
			log.Printf("DEBUG: Cache disabled, loading via packages.Load...")
		} else {
			log.Printf("DEBUG: Cache miss or invalid for key: %s. Loading via packages.Load...", string(cacheKey))
		}

		// 1. Load Package Info
		loadStart := time.Now()
		fset := token.NewFileSet()
		targetPkg, targetFileAST, targetFile, loadErrors := loadPackageInfo(ctx, absFilename, fset)
		loadDuration = time.Since(loadStart)
		log.Printf("DEBUG: packages.Load completed in %v", loadDuration)
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr)
		}

		// 2. Perform Detailed Analysis Steps
		if targetFile != nil && fset != nil {
			stepsStart := time.Now()
			analyzeStepErr := a.performAnalysisSteps(targetFile, targetFileAST, targetPkg, fset, line, col, info)
			stepsDuration = time.Since(stepsStart)
			log.Printf("DEBUG: performAnalysisSteps completed in %v", stepsDuration)
			if analyzeStepErr != nil {
				addAnalysisError(info, analyzeStepErr)
			}
		} else {
			if targetFile == nil {
				addAnalysisError(info, errors.New("cannot perform analysis steps: missing target file after load"))
			}
			if fset == nil && targetFile != nil {
				addAnalysisError(info, errors.New("cannot perform analysis steps: missing FileSet after load"))
			}
		}

		// 3. Build Preamble
		var qualifier types.Qualifier
		if targetPkg != nil && targetPkg.Types != nil {
			info.PackageName = targetPkg.Types.Name()
			qualifier = types.RelativeTo(targetPkg.Types)
		} else {
			qualifier = func(other *types.Package) string {
				if other != nil {
					return other.Path()
				}
				return ""
			}
			log.Printf("DEBUG: Building preamble with limited/no type info.")
		}
		preambleStart := time.Now()
		info.PromptPreamble = buildPreamble(info, qualifier)
		preambleDuration = time.Since(preambleStart)
		log.Printf("DEBUG: buildPreamble completed in %v", preambleDuration)

		// 4. Save to Cache
		if a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0 { // Only cache if load was clean
			log.Printf("DEBUG: Attempting to save analysis results to bbolt cache. Key: %s", string(cacheKey))
			saveStart := time.Now()
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg)
			if hashErr == nil {
				analysisDataToCache := CachedAnalysisData{
					PackageName:    info.PackageName,
					PromptPreamble: info.PromptPreamble,
				}
				var gobBuf bytes.Buffer
				encoder := gob.NewEncoder(&gobBuf)
				if encodeErr := encoder.Encode(&analysisDataToCache); encodeErr == nil {
					analysisGob := gobBuf.Bytes()
					entryToSave := CachedAnalysisEntry{
						SchemaVersion:   cacheSchemaVersion,
						GoModHash:       goModHash,
						InputFileHashes: inputHashes,
						AnalysisGob:     analysisGob,
					}
					var entryBuf bytes.Buffer
					entryEncoder := gob.NewEncoder(&entryBuf)
					if entryEncodeErr := entryEncoder.Encode(&entryToSave); entryEncodeErr == nil {
						encodedBytes := entryBuf.Bytes()
						saveErr := a.db.Update(func(tx *bbolt.Tx) error {
							b := tx.Bucket(cacheBucketName)
							if b == nil {
								return fmt.Errorf("%w: cache bucket %s disappeared during save", ErrCache, string(cacheBucketName))
							}
							log.Printf("DEBUG: Writing %d bytes to cache for key %s", len(encodedBytes), string(cacheKey))
							return b.Put(cacheKey, encodedBytes)
						})
						if saveErr == nil {
							log.Printf("DEBUG: Saved analysis results to bbolt cache %s in %v", string(cacheKey), time.Since(saveStart))
						} else {
							log.Printf("Warning: Failed to write to bbolt cache for key %s: %v", string(cacheKey), saveErr)
							addAnalysisError(info, fmt.Errorf("%w: write error: %w", ErrCache, saveErr))
						}
					} else {
						log.Printf("Warning: Failed to gob-encode cache entry for key %s: %v", string(cacheKey), entryEncodeErr)
						addAnalysisError(info, fmt.Errorf("%w: entry encode error: %w", ErrCache, entryEncodeErr))
					}
				} else {
					log.Printf("Warning: Failed to gob-encode analysis data for caching: %v", encodeErr)
					addAnalysisError(info, fmt.Errorf("%w: data encode error: %w", ErrCache, encodeErr))
				}
			} else {
				log.Printf("Warning: Failed to calculate input hashes for caching: %v", hashErr)
				addAnalysisError(info, fmt.Errorf("%w: hash calculation error: %w", ErrCache, hashErr))
			}
		} else if a.db != nil {
			log.Printf("DEBUG: Skipping cache save for key %s (Load Errors: %d, Preamble Empty: %t)",
				string(cacheKey), len(loadErrors), info.PromptPreamble == "")
		}
	}

	// Log final state
	logAnalysisErrors(info.AnalysisErrors)
	analysisErr = errors.Join(info.AnalysisErrors...)

	if cacheHit {
		log.Printf("Context analysis finished (Decode: %v, Cache Hit: %t)", loadDuration, cacheHit)
	} else {
		log.Printf("Context analysis finished (Load: %v, Steps: %v, Preamble: %v, Cache Hit: %t)",
			loadDuration, stepsDuration, preambleDuration, cacheHit)
	}
	if len(info.PromptPreamble) > 500 {
		log.Printf("Final Context Preamble (length %d):\n---\n%s\n... (preamble truncated in log)\n---", len(info.PromptPreamble), info.PromptPreamble[:500])
	} else {
		log.Printf("Final Context Preamble (length %d):\n---\n%s\n---", len(info.PromptPreamble), info.PromptPreamble)
	}

	// Determine final return error
	if len(info.AnalysisErrors) > 0 && info.PromptPreamble == "" {
		return info, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisErr) // Fatal if errors and no preamble
	} else if len(info.AnalysisErrors) > 0 {
		return info, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisErr) // Non-fatal if errors but preamble exists
	}
	return info, nil // Success
}

// --- InvalidateCache Method ---
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	a.mu.Lock()
	db := a.db
	a.mu.Unlock()
	if db == nil {
		log.Printf("DEBUG: Cache invalidation skipped: DB is nil.")
		return nil
	}

	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	log.Printf("DEBUG: Invalidating cache for key: %s", string(cacheKey))
	err := db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(cacheBucketName)
		if b == nil {
			log.Printf("Warning: Cache bucket %s not found during invalidation.", string(cacheBucketName))
			return nil
		}
		if b.Get(cacheKey) == nil {
			log.Printf("DEBUG: Cache entry for key %s already deleted or never existed.", string(cacheKey))
			return nil
		}
		log.Printf("DEBUG: Deleting cache entry for key: %s", string(cacheKey))
		return b.Delete(cacheKey)
	})
	if err != nil {
		log.Printf("Warning: Failed to delete cache entry %s: %v", string(cacheKey), err)
		return fmt.Errorf("%w: failed to delete entry %s: %w", ErrCache, string(cacheKey), err)
	}
	log.Printf("DEBUG: Successfully invalidated cache entry for key: %s", string(cacheKey))
	return nil
}

// performAnalysisSteps encapsulates the core analysis logic after loading/parsing.
func (a *GoPackagesAnalyzer) performAnalysisSteps(
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package,
	fset *token.FileSet,
	line, col int,
	info *AstContextInfo,
) error {
	if targetFile == nil || fset == nil {
		return errors.New("performAnalysisSteps requires non-nil targetFile and fset")
	}

	cursorPos, posErr := calculateCursorPos(targetFile, line, col)
	if posErr != nil {
		return fmt.Errorf("cursor position error: %w", posErr)
	}
	info.CursorPos = cursorPos
	log.Printf("Calculated cursor token.Pos: %d (%s)", info.CursorPos, fset.PositionFor(info.CursorPos, true))

	if targetFileAST != nil {
		path := findEnclosingPath(targetFileAST, info.CursorPos, info)
		findContextNodes(path, info.CursorPos, targetPkg, fset, info)
		gatherScopeContext(path, targetPkg, fset, info)
		findRelevantComments(targetFileAST, path, info.CursorPos, fset, info)
	} else {
		addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil"))
		gatherScopeContext(nil, targetPkg, fset, info) // Still try to get package scope
	}
	return nil
}

// --- Prompt Formatter ---
type templateFormatter struct{}

func newTemplateFormatter() *templateFormatter { return &templateFormatter{} }

func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2

	// Truncate preamble (keep end)
	if len(contextPreamble) > maxPreambleLen {
		log.Printf("Warning: Truncating context preamble from %d to %d bytes.", len(contextPreamble), maxPreambleLen)
		startByte := len(contextPreamble) - maxPreambleLen + len("... (context truncated)\n")
		if startByte < 0 {
			startByte = 0
		}
		contextPreamble = "... (context truncated)\n" + contextPreamble[startByte:]
	}

	if config.UseFim {
		template = config.FimTemplate
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix

		// Truncate FIM prefix (keep end)
		if len(prefix) > maxFIMPartLen {
			log.Printf("Warning: Truncating FIM prefix from %d to %d bytes.", len(prefix), maxFIMPartLen)
			startByte := len(prefix) - maxFIMPartLen + len("...(prefix truncated)")
			if startByte < 0 {
				startByte = 0
			}
			prefix = "...(prefix truncated)" + prefix[startByte:]
		}
		// Truncate FIM suffix (keep start)
		if len(suffix) > maxFIMPartLen {
			log.Printf("Warning: Truncating FIM suffix from %d to %d bytes.", len(suffix), maxFIMPartLen)
			endByte := maxFIMPartLen - len("(suffix truncated)...")
			if endByte < 0 {
				endByte = 0
			}
			suffix = suffix[:endByte] + "(suffix truncated)..."
		}
		finalPrompt = fmt.Sprintf(template, contextPreamble, prefix, suffix)
	} else { // Standard completion
		snippet := snippetCtx.Prefix
		// Truncate snippet (keep end)
		if len(snippet) > maxSnippetLen {
			log.Printf("Warning: Truncating code snippet from %d to %d bytes.", len(snippet), maxSnippetLen)
			startByte := len(snippet) - maxSnippetLen + len("...(code truncated)\n")
			if startByte < 0 {
				startByte = 0
			}
			snippet = "...(code truncated)\n" + snippet[startByte:]
		}
		finalPrompt = fmt.Sprintf(template, contextPreamble, snippet)
	}
	return finalPrompt
}

// =============================================================================
// DeepCompleter Service
// =============================================================================
type DeepCompleter struct {
	client    LLMClient
	analyzer  Analyzer
	formatter PromptFormatter
	config    Config
}

func NewDeepCompleter() (*DeepCompleter, error) {
	cfg, configErr := LoadConfig()
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		return nil, configErr
	} // Fatal config error
	if configErr != nil {
		log.Printf("Warning during config load: %v", configErr)
	} // Non-fatal config error

	// Ensure config is valid even if load had non-fatal errors
	if err := cfg.Validate(); err != nil {
		log.Printf("Warning: Config after load/merge is invalid: %v. Using pure defaults.", err)
		cfg = DefaultConfig
		if valErr := cfg.Validate(); valErr != nil {
			return nil, fmt.Errorf("default config validation failed: %w", valErr)
		}
	}

	analyzer := NewGoPackagesAnalyzer()
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    cfg,
	}
	// Return non-fatal errors from LoadConfig if any occurred
	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}

func NewDeepCompleterWithConfig(config Config) (*DeepCompleter, error) {
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	analyzer := NewGoPackagesAnalyzer()
	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    config,
	}, nil
}

func (dc *DeepCompleter) Close() error {
	if dc.analyzer != nil {
		return dc.analyzer.Close()
	}
	return nil
}

func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	log.Println("DeepCompleter.GetCompletion called for basic prompt.")
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, dc.config)

	reader, err := dc.client.GenerateStream(ctx, prompt, dc.config)
	if err != nil {
		if errors.Is(err, ErrOllamaUnavailable) {
			return "", err
		}
		return "", fmt.Errorf("failed to call Ollama API for basic prompt: %w", err)
	}
	var buffer bytes.Buffer
	streamCtx, cancelStream := context.WithTimeout(ctx, 50*time.Second)
	defer cancelStream()
	if streamErr := streamCompletion(streamCtx, reader, &buffer); streamErr != nil {
		if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
			return "", fmt.Errorf("%w: streaming context error: %w", ErrOllamaUnavailable, streamErr)
		}
		return "", fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
	}
	return strings.TrimSpace(buffer.String()), nil
}

func (dc *DeepCompleter) GetCompletionStreamFromFile(
	ctx context.Context,
	filename string,
	row, col int,
	w io.Writer,
) error {
	var contextPreamble string = "// Basic file context only."
	var analysisInfo *AstContextInfo
	var analysisErr error

	if dc.config.UseAst {
		log.Printf("Analyzing context (or checking cache) for %s:%d:%d", filename, row, col)
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second)
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, filename, row, col)
		cancelAnalysis()

		// Handle fatal analysis errors
		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) && !errors.Is(analysisErr, ErrCache) {
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		// Log non-fatal errors
		if analysisErr != nil && analysisInfo != nil {
			logAnalysisErrors(analysisInfo.AnalysisErrors)
			log.Printf("Non-fatal error during analysis/cache check: %v", analysisErr)
		}

		// Use preamble from analysisInfo (could be from cache or fresh analysis)
		if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else if analysisErr != nil {
			contextPreamble += fmt.Sprintf("\n// Warning: Context analysis completed with errors and no preamble: %v\n", analysisErr)
		} else {
			contextPreamble += "\n// Warning: Context analysis returned no specific context preamble.\n"
		}
	} else {
		log.Println("AST analysis disabled by config.")
	}

	// Extract code snippet around the cursor
	snippetCtx, err := extractSnippetContext(filename, row, col) // Restored call
	if err != nil {
		return fmt.Errorf("failed to extract code snippet context: %w", err)
	}

	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, dc.config)
	if len(prompt) > 1000 {
		log.Printf("Generated Prompt (length %d):\n---\n%s\n... (prompt truncated in log)\n---", len(prompt), prompt[:1000])
	} else {
		log.Printf("Generated Prompt (length %d):\n---\n%s\n---", len(prompt), prompt)
	}

	// Retryable API call function
	apiCallFunc := func() error {
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, dc.config)
		if apiErr != nil {
			var oe *OllamaError
			isRetryable := errors.As(apiErr, &oe) &&
				(oe.Status == http.StatusServiceUnavailable || oe.Status == http.StatusTooManyRequests)
			isRetryable = isRetryable || errors.Is(apiErr, context.DeadlineExceeded)
			if isRetryable {
				return apiErr
			}
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, apiErr)
		}
		PrettyPrint(ColorGreen, "Completion:\n")
		streamErr := streamCompletion(apiCtx, reader, w)
		fmt.Fprintln(w) // Add newline after stream
		if streamErr != nil {
			if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
				return streamErr
			}
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		log.Println("Completion stream finished successfully for this attempt.")
		return nil
	}

	// Execute with retry logic
	err = retry(ctx, apiCallFunc, maxRetries, retryDelay)
	if err != nil {
		var oe *OllamaError
		if errors.As(err, &oe) || errors.Is(err, ErrOllamaUnavailable) ||
			errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			return err
		}
		return fmt.Errorf("failed to get completion stream after retries: %w", err)
	}

	// Log analysis errors again as a final warning
	if analysisErr != nil {
		log.Printf("Warning: Completion succeeded, but context analysis/cache check encountered non-fatal errors: %v", analysisErr)
	}
	return nil // Success
}

// =============================================================================
// Internal Helpers (Unexported package-level functions)
// =============================================================================

// --- Ollama Stream Processing Helpers ---
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer) error {
	defer r.Close()
	reader := bufio.NewReader(r)
	for {
		select {
		case <-ctx.Done():
			log.Println("Context cancelled during streaming")
			return ctx.Err()
		default:
		}
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if len(line) > 0 {
					if procErr := processLine(line, w); procErr != nil {
						return procErr
					}
				}
				return nil // Successful end of stream
			}
			select { // Check if error is due to context cancellation
			case <-ctx.Done():
				return ctx.Err()
			default:
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			}
		}
		if procErr := processLine(line, w); procErr != nil {
			return procErr
		}
	}
}

func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil
	}
	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		log.Printf("Debug: Ignoring non-JSON line from Ollama stream: %s", string(line))
		return nil // Tolerate non-JSON lines
	}
	if resp.Error != "" {
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	}
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		return fmt.Errorf("error writing to output: %w", err)
	}
	return nil
}

// --- Retry Helper ---
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration) error {
	var lastErr error
	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled before attempt %d: %v", i+1, ctx.Err())
			return ctx.Err()
		default:
		}

		lastErr = operation()
		if lastErr == nil {
			return nil
		} // Success

		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			log.Printf("Attempt %d failed due to context error: %v. Not retrying.", i+1, lastErr)
			return lastErr
		}
		var ollamaErr *OllamaError
		isRetryable := errors.As(lastErr, &ollamaErr) &&
			(ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		isRetryable = isRetryable || errors.Is(lastErr, ErrOllamaUnavailable)

		if !isRetryable {
			log.Printf("Attempt %d failed with non-retryable error: %v", i+1, lastErr)
			return lastErr
		}

		waitDuration := currentDelay
		log.Printf("Attempt %d failed with retryable error: %v. Retrying in %v...", i+1, lastErr, waitDuration)
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled during retry wait: %v", ctx.Err())
			return ctx.Err()
		case <-time.After(waitDuration):
			// Consider exponential backoff: currentDelay *= 2
		}
	}
	log.Printf("Operation failed after %d retries.", maxRetries)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr)
}

// --- Analysis Helpers ---
func loadPackageInfo(ctx context.Context, absFilename string, fset *token.FileSet) (*packages.Package, *ast.File, *token.File, []error) {
	dir := filepath.Dir(absFilename)
	var loadErrors []error
	cfg := &packages.Config{
		Context: ctx,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedSyntax |
			packages.NeedTypesInfo | packages.NeedTypesSizes | packages.NeedDeps,
		Dir:  dir,
		Fset: fset,
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			const mode = parser.ParseComments | parser.AllErrors
			file, err := parser.ParseFile(fset, filename, src, mode)
			if err != nil {
				log.Printf("Parser error in %s (proceeding with partial AST): %v", filename, err)
			}
			return file, nil
		},
		Tests: true,
	}
	pkgs, loadErr := packages.Load(cfg, ".")
	if loadErr != nil {
		log.Printf("Error loading packages for %s: %v", dir, loadErr)
		loadErrors = append(loadErrors, fmt.Errorf("package loading failed: %w", loadErr))
	}
	if len(pkgs) > 0 {
		packages.Visit(pkgs, nil, func(pkg *packages.Package) {
			for _, err := range pkg.Errors {
				errMsg := err.Error()
				found := false
				for _, existing := range loadErrors {
					if existing.Error() == errMsg {
						found = true
						break
					}
				}
				if !found {
					loadErrors = append(loadErrors, err)
				}
			}
		})
		if len(loadErrors) > 0 {
			log.Printf("Detailed errors encountered during package loading for %s", dir)
		}
	} else if loadErr == nil {
		loadErrors = append(loadErrors, fmt.Errorf("no packages found in directory %s", dir))
	}
	for _, pkg := range pkgs {
		if pkg == nil || pkg.Fset == nil || pkg.Syntax == nil {
			continue
		}
		for _, syntaxFile := range pkg.Syntax {
			if syntaxFile == nil {
				continue
			}
			tokenFile := pkg.Fset.File(syntaxFile.Pos())
			if tokenFile != nil && tokenFile.Name() != "" {
				normSyntaxFileName, _ := filepath.Abs(tokenFile.Name())
				if normSyntaxFileName == absFilename {
					if pkg.TypesInfo == nil {
						log.Printf("Warning: Type info missing for target package %s", pkg.PkgPath)
						loadErrors = append(loadErrors, fmt.Errorf("type info missing for package %s", pkg.PkgPath))
					}
					if pkg.Types == nil {
						log.Printf("Warning: Types definition missing for target package %s", pkg.PkgPath)
						loadErrors = append(loadErrors, fmt.Errorf("types definition missing for package %s", pkg.PkgPath))
					}
					return pkg, syntaxFile, tokenFile, loadErrors
				}
			}
		}
	}
	if len(loadErrors) == 0 {
		loadErrors = append(loadErrors, fmt.Errorf("target file %s not found within loaded packages for directory %s", absFilename, dir))
	}
	return nil, nil, nil, loadErrors
}

func findEnclosingPath(targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo) []ast.Node {
	if targetFileAST == nil {
		addAnalysisError(info, errors.New("cannot find enclosing path: targetFileAST is nil"))
		return nil
	}
	if !cursorPos.IsValid() {
		addAnalysisError(info, errors.New("cannot find enclosing path: invalid cursor position"))
		return nil
	}
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		log.Printf("DEBUG: No AST path found enclosing cursor position %v", cursorPos)
	}
	return path
}

func gatherScopeContext(path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo) {
	// Removed unused qualifier variable declaration
	if fset == nil && path != nil {
		log.Println("Warning: Cannot format receiver type - fset is nil in gatherScopeContext.")
	}

	if path != nil {
		for i := len(path) - 1; i >= 0; i-- {
			node := path[i]
			switch n := node.(type) {
			case *ast.FuncDecl:
				if info.EnclosingFuncNode == nil {
					info.EnclosingFuncNode = n
					if fset != nil && n.Recv != nil && len(n.Recv.List) > 0 && n.Recv.List[0].Type != nil {
						var buf bytes.Buffer
						if err := format.Node(&buf, fset, n.Recv.List[0].Type); err == nil {
							info.ReceiverType = buf.String()
						} else {
							log.Printf("Warning: could not format receiver type: %v", err)
							info.ReceiverType = "[error formatting receiver]"
							addAnalysisError(info, fmt.Errorf("receiver format error: %w", err))
						}
					}
				}
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
					if obj, ok := targetPkg.TypesInfo.Defs[n.Name]; ok && obj != nil {
						if fn, ok := obj.(*types.Func); ok {
							if info.EnclosingFunc == nil {
								info.EnclosingFunc = fn
								if sig, ok := fn.Type().(*types.Signature); ok {
									addSignatureToScope(sig, info.VariablesInScope)
								}
							}
						}
					} else {
						if info.EnclosingFunc == nil && n.Name != nil {
							addAnalysisError(info, fmt.Errorf("definition for func '%s' not found in TypesInfo", n.Name.Name))
						}
					}
				} else if info.EnclosingFuncNode != nil && info.EnclosingFunc == nil {
					reason := "type info unavailable"
					if targetPkg != nil && targetPkg.TypesInfo == nil {
						reason = "TypesInfo is nil"
					}
					if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs == nil {
						reason = "TypesInfo.Defs is nil"
					}
					funcName := "[anonymous]"
					if info.EnclosingFuncNode.Name != nil {
						funcName = info.EnclosingFuncNode.Name.Name
					}
					addAnalysisError(info, fmt.Errorf("type info for enclosing func '%s' unavailable: %s", funcName, reason))
				}
			case *ast.BlockStmt:
				if info.EnclosingBlock == nil {
					info.EnclosingBlock = n
				}
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					}
				}
			}
		}
	}
	addPackageScope(targetPkg, info)
}

func addPackageScope(targetPkg *packages.Package, info *AstContextInfo) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope)
		} else {
			addAnalysisError(info, fmt.Errorf("package scope missing for pkg %s", targetPkg.PkgPath))
		}
	} else {
		if targetPkg != nil {
			addAnalysisError(info, fmt.Errorf("package.Types field is nil for pkg %s", targetPkg.PkgPath))
		} else {
			addAnalysisError(info, errors.New("cannot add package scope: targetPkg is nil"))
		}
	}
}

func addScopeVariables(typeScope *types.Scope, cursorPos token.Pos, scopeMap map[string]types.Object) {
	if typeScope == nil {
		return
	}
	for _, name := range typeScope.Names() {
		obj := typeScope.Lookup(name)
		if obj == nil {
			continue
		}
		include := !cursorPos.IsValid() || !obj.Pos().IsValid() || obj.Pos() < cursorPos
		if include {
			if _, exists := scopeMap[name]; !exists {
				switch obj.(type) {
				case *types.Var, *types.Const, *types.TypeName, *types.Func, *types.Label, *types.PkgName, *types.Builtin, *types.Nil:
					scopeMap[name] = obj
				}
			}
		}
	}
}

func addSignatureToScope(sig *types.Signature, scopeMap map[string]types.Object) {
	if sig == nil {
		return
	}
	addTupleToScope(sig.Params(), scopeMap)
	addTupleToScope(sig.Results(), scopeMap)
}

func addTupleToScope(tuple *types.Tuple, scopeMap map[string]types.Object) {
	if tuple == nil {
		return
	}
	for j := 0; j < tuple.Len(); j++ {
		v := tuple.At(j)
		if v != nil && v.Name() != "" {
			if _, exists := scopeMap[v.Name()]; !exists {
				scopeMap[v.Name()] = v
			}
		}
	}
}

func findRelevantComments(targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo) {
	if targetFileAST == nil || fset == nil {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"))
		return
	}
	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset)
}

func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		return comments
	}

	cursorLine := fset.Position(cursorPos).Line
	foundPreceding := false
	var precedingComments []string

	// Strategy 1: Find comments ending on the line immediately before the cursor
	for node := range cmap {
		if node == nil {
			continue
		}
		for _, cg := range cmap[node] {
			if cg == nil {
				continue
			}
			if cg.End().IsValid() && fset.Position(cg.End()).Line == cursorLine-1 {
				for _, c := range cg.List {
					if c != nil {
						precedingComments = append(precedingComments, c.Text)
					}
				}
				foundPreceding = true
				break
			}
		}
		if foundPreceding {
			break
		}
	}

	if foundPreceding {
		comments = append(comments, precedingComments...)
	} else {
		// Strategy 2: If no preceding comment found, look for Doc comments on enclosing path
		if path != nil {
			for i := 0; i < len(path); i++ {
				node := path[i]
				var docComment *ast.CommentGroup
				switch n := node.(type) {
				case *ast.FuncDecl:
					docComment = n.Doc
				case *ast.GenDecl:
					docComment = n.Doc
				case *ast.TypeSpec:
					docComment = n.Doc
				case *ast.Field:
					docComment = n.Doc
				case *ast.ValueSpec:
					docComment = n.Doc
				}
				if docComment != nil {
					for _, c := range docComment.List {
						if c != nil {
							comments = append(comments, c.Text)
						}
					}
					goto cleanup
				}
			}
		}
	}

cleanup: // Deduplicate final list
	seen := make(map[string]struct{})
	uniqueComments := make([]string, 0, len(comments))
	for _, c := range comments {
		if _, ok := seen[c]; !ok {
			seen[c] = struct{}{}
			uniqueComments = append(uniqueComments, c)
		}
	}
	return uniqueComments
}

func buildPreamble(info *AstContextInfo, qualifier types.Qualifier) string {
	var preamble strings.Builder
	const internalPreambleLimit = 8192
	currentLen := 0
	addToPreamble := func(s string) bool {
		if currentLen+len(s) < internalPreambleLimit {
			preamble.WriteString(s)
			currentLen += len(s)
			return true
		}
		return false
	}
	addTruncMarker := func(section string) {
		msg := fmt.Sprintf("//   ... (%s truncated)\n", section)
		if currentLen+len(msg) < internalPreambleLimit {
			preamble.WriteString(msg)
			currentLen += len(msg)
		}
	}

	if !addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), info.PackageName)) {
		return preamble.String()
	}
	if !formatImportsSection(&preamble, info, addToPreamble, addTruncMarker) {
		return preamble.String()
	}
	if !formatEnclosingFuncSection(&preamble, info, qualifier, addToPreamble) {
		return preamble.String()
	}
	if !formatCommentsSection(&preamble, info, addToPreamble, addTruncMarker) {
		return preamble.String()
	}
	if !formatCursorContextSection(&preamble, info, qualifier, addToPreamble) {
		return preamble.String()
	}
	formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker)
	return preamble.String()
}

func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string)) bool {
	if len(info.Imports) == 0 {
		return true
	}
	header := "// Imports:\n"
	if !add(header) {
		return false
	}
	count := 0
	maxImports := 20
	for _, imp := range info.Imports {
		if imp == nil || imp.Path == nil {
			continue
		}
		if count >= maxImports {
			addTrunc("imports")
			return true
		}
		path := imp.Path.Value
		name := ""
		if imp.Name != nil {
			name = imp.Name.Name + " "
		}
		line := fmt.Sprintf("//   import %s%s\n", name, path)
		if !add(line) {
			return false
		}
		count++
	}
	return true
}

func formatEnclosingFuncSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool) bool {
	funcOrMethod := "Function"
	receiverStr := ""
	if info.ReceiverType != "" {
		funcOrMethod = "Method"
		receiverStr = fmt.Sprintf("(%s) ", info.ReceiverType)
	}
	var funcHeader string
	if info.EnclosingFunc != nil {
		name := info.EnclosingFunc.Name()
		sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		if info.ReceiverType != "" && strings.HasPrefix(sigStr, "func(") {
			sigStr = "func" + strings.TrimPrefix(sigStr, "func")
		}
		funcHeader = fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr)
	} else if info.EnclosingFuncNode != nil {
		name := "[anonymous]"
		if info.EnclosingFuncNode.Name != nil {
			name = info.EnclosingFuncNode.Name.Name
		}
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s(...)\n", funcOrMethod, receiverStr, name)
	} else {
		return true
	}
	return add(funcHeader)
}

func formatCommentsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string)) bool {
	if len(info.CommentsNearCursor) == 0 {
		return true
	}
	header := "// Relevant Comments:\n"
	if !add(header) {
		return false
	}
	count := 0
	maxComments := 5
	for _, c := range info.CommentsNearCursor {
		if count >= maxComments {
			addTrunc("comments")
			return true
		}
		cleanComment := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(strings.TrimPrefix(c, "//")), "*/")), "/*"))
		if len(cleanComment) > 0 {
			line := fmt.Sprintf("//   %s\n", cleanComment)
			if !add(line) {
				return false
			}
			count++
		}
	}
	return true
}

func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool) bool {
	if info.CallExpr != nil {
		funcName := "[unknown function]"
		switch fun := info.CallExpr.Fun.(type) {
		case *ast.Ident:
			funcName = fun.Name
		case *ast.SelectorExpr:
			if fun.Sel != nil {
				funcName = fun.Sel.Name
			}
		}
		if !add(fmt.Sprintf("// Inside function call: %s (Arg %d)\n", funcName, info.CallArgIndex+1)) {
			return false
		}
		if sig := info.CallExprFuncType; sig != nil {
			if !add(fmt.Sprintf("// Function Signature: %s\n", types.TypeString(sig, qualifier))) {
				return false
			}
			params := sig.Params()
			if params != nil && params.Len() > 0 {
				if !add("//   Parameters:\n") {
					return false
				}
				for i := 0; i < params.Len(); i++ {
					p := params.At(i)
					if p == nil {
						continue
					}
					highlight := ""
					if i == info.CallArgIndex {
						highlight = " // <-- cursor here"
					}
					if sig.Variadic() && i == params.Len()-1 && info.CallArgIndex >= i {
						highlight = " // <-- cursor here (variadic)"
					}
					if !add(fmt.Sprintf("//     - %s%s\n", types.ObjectString(p, qualifier), highlight)) {
						return false
					}
				}
			} else {
				if !add("//   Parameters: (none)\n") {
					return false
				}
			}
			results := sig.Results()
			if results != nil && results.Len() > 0 {
				if !add("//   Returns:\n") {
					return false
				}
				for i := 0; i < results.Len(); i++ {
					r := results.At(i)
					if r == nil {
						continue
					}
					if !add(fmt.Sprintf("//     - %s\n", types.ObjectString(r, qualifier))) {
						return false
					}
				}
			} else {
				if !add("//   Returns: (none)\n") {
					return false
				}
			}
		} else {
			if !add("// Function Signature: (unknown - type analysis failed for call expression)\n") {
				return false
			}
		}
		return true
	}
	if info.SelectorExpr != nil {
		selName := ""
		if info.SelectorExpr.Sel != nil {
			selName = info.SelectorExpr.Sel.Name
		}
		typeName := "(unknown - type analysis failed for base expression)"
		if info.SelectorExprType != nil {
			typeName = types.TypeString(info.SelectorExprType, qualifier)
		}
		if !add(fmt.Sprintf("// Selector context: expr type = %s (selecting '%s')\n", typeName, selName)) {
			return false
		}
		if info.SelectorExprType != nil {
			members := listTypeMembers(info.SelectorExprType, info.SelectorExpr.X, qualifier)
			var fields, methods []MemberInfo
			if members != nil {
				for _, member := range members {
					if member.Kind == FieldMember {
						fields = append(fields, member)
					}
					if member.Kind == MethodMember {
						methods = append(methods, member)
					}
				}
			}
			if len(fields) > 0 {
				if !add("//   Available Fields:\n") {
					return false
				}
				sort.Slice(fields, func(i, j int) bool { return fields[i].Name < fields[j].Name })
				for _, field := range fields {
					if !add(fmt.Sprintf("//     - %s %s\n", field.Name, field.TypeString)) {
						return false
					}
				}
			}
			if len(methods) > 0 {
				if !add("//   Available Methods:\n") {
					return false
				}
				sort.Slice(methods, func(i, j int) bool { return methods[i].Name < methods[j].Name })
				for _, method := range methods {
					methodSig := strings.TrimPrefix(method.TypeString, "func")
					if !add(fmt.Sprintf("//     - %s%s\n", method.Name, methodSig)) {
						return false
					}
				}
			}
			if len(fields) == 0 && len(methods) == 0 {
				if members == nil {
					if !add("//   (Could not determine members)\n") {
						return false
					}
				} else {
					if !add("//   (No exported fields or methods found)\n") {
						return false
					}
				}
			}
		} else {
			if !add("//   (Cannot list members: type analysis failed for base expression)\n") {
				return false
			}
		}
		return true
	}
	if info.CompositeLit != nil {
		typeName := "(unknown - type analysis failed for literal)"
		if info.CompositeLitType != nil {
			typeName = types.TypeString(info.CompositeLitType, qualifier)
		}
		if !add(fmt.Sprintf("// Inside composite literal of type: %s\n", typeName)) {
			return false
		}
		if info.CompositeLitType != nil {
			var st *types.Struct
			currentType := info.CompositeLitType.Underlying()
			if ptr, ok := currentType.(*types.Pointer); ok {
				if ptr.Elem() != nil {
					currentType = ptr.Elem().Underlying()
				} else {
					currentType = nil
				}
			}
			st, _ = currentType.(*types.Struct)
			if st != nil {
				presentFields := make(map[string]bool)
				for _, elt := range info.CompositeLit.Elts {
					if kv, ok := elt.(*ast.KeyValueExpr); ok {
						if kid, ok := kv.Key.(*ast.Ident); ok {
							presentFields[kid.Name] = true
						}
					}
				}
				var missingFields []string
				for i := 0; i < st.NumFields(); i++ {
					field := st.Field(i)
					if field != nil && field.Exported() && !presentFields[field.Name()] {
						missingFields = append(missingFields, types.ObjectString(field, qualifier))
					}
				}
				if len(missingFields) > 0 {
					if !add("//   Missing Exported Fields (candidates for completion):\n") {
						return false
					}
					sort.Strings(missingFields)
					for _, fieldStr := range missingFields {
						if !add(fmt.Sprintf("//     - %s\n", fieldStr)) {
							return false
						}
					}
				} else {
					if !add("//   (All exported fields may be present or none missing)\n") {
						return false
					}
				}
			} else {
				if !add("//   (Underlying type is not a struct)\n") {
					return false
				}
			}
		} else {
			if !add("//   (Cannot determine missing fields: type analysis failed)\n") {
				return false
			}
		}
		return true
	}
	if info.IdentifierAtCursor != nil {
		identName := info.IdentifierAtCursor.Name
		if info.IdentifierType != nil {
			if !add(fmt.Sprintf("// Identifier at cursor: %s (Type: %s)\n", identName, types.TypeString(info.IdentifierType, qualifier))) {
				return false
			}
		} else {
			if !add(fmt.Sprintf("// Identifier at cursor: %s (Type unknown)\n", identName)) {
				return false
			}
		}
		return true
	}
	return true
}

func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string)) bool {
	if len(info.VariablesInScope) == 0 {
		return true
	}
	header := "// Variables/Constants/Types in Scope:\n"
	if !add(header) {
		return false
	}
	var items []string
	for name := range info.VariablesInScope {
		obj := info.VariablesInScope[name]
		items = append(items, fmt.Sprintf("//   %s\n", types.ObjectString(obj, qualifier)))
	}
	sort.Strings(items)
	count := 0
	maxScopeItems := 30
	for _, item := range items {
		if count >= maxScopeItems {
			addTrunc("scope")
			return true
		}
		if !add(item) {
			return false
		}
		count++
	}
	return true
}

func calculateCursorPos(file *token.File, line, col int) (token.Pos, error) {
	if line <= 0 {
		return token.NoPos, fmt.Errorf("invalid line number: %d (must be >= 1)", line)
	}
	if col <= 0 {
		return token.NoPos, fmt.Errorf("invalid column number: %d (must be >= 1)", col)
	}
	if file == nil {
		return token.NoPos, errors.New("invalid token.File (nil)")
	}

	fileLineCount := file.LineCount()
	if line > fileLineCount {
		if line == fileLineCount+1 && col == 1 {
			return file.Pos(file.Size()), nil
		}
		return token.NoPos, fmt.Errorf("line number %d exceeds file line count %d", line, fileLineCount)
	}
	lineStartPos := file.LineStart(line)
	if !lineStartPos.IsValid() {
		return token.NoPos, fmt.Errorf("cannot get start offset for line %d in file '%s'", line, file.Name())
	}

	lineStartOffset := file.Offset(lineStartPos)
	cursorOffset := lineStartOffset + col - 1
	lineEndOffset := file.Size()
	if line < fileLineCount {
		nextLineStartPos := file.LineStart(line + 1)
		if nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	finalOffset := cursorOffset
	if cursorOffset < lineStartOffset {
		log.Printf("Warning: column %d resulted in offset %d before line start %d. Clamping to line start.", col, cursorOffset, lineStartOffset)
		finalOffset = lineStartOffset
	} else if cursorOffset > lineEndOffset {
		log.Printf("Warning: column %d resulted in offset %d beyond line end %d. Clamping to line end.", col, cursorOffset, lineEndOffset)
		finalOffset = lineEndOffset
	}

	pos := file.Pos(finalOffset)
	if !pos.IsValid() {
		log.Printf("Error: Clamped offset %d resulted in invalid token.Pos. Using line start %d.", finalOffset, lineStartPos)
		return lineStartPos, fmt.Errorf("failed to calculate valid token.Pos for offset %d", finalOffset)
	}
	return pos, nil
}

func findContextNodes(path []ast.Node, cursorPos token.Pos, pkg *packages.Package, fset *token.FileSet, info *AstContextInfo) {
	if len(path) == 0 || fset == nil {
		if fset == nil {
			addAnalysisError(info, errors.New("Fset is nil in findContextNodes"))
		}
		return
	}
	posStr := func(p token.Pos) string {
		if p.IsValid() {
			return fset.Position(p).String()
		}
		return fmt.Sprintf("Pos(%d)", p)
	}

	hasTypeInfo := pkg != nil && pkg.TypesInfo != nil
	var typesMap map[ast.Expr]types.TypeAndValue
	var defsMap map[*ast.Ident]types.Object
	var usesMap map[*ast.Ident]types.Object
	if hasTypeInfo {
		typesMap = pkg.TypesInfo.Types
		defsMap = pkg.TypesInfo.Defs
		usesMap = pkg.TypesInfo.Uses
		if typesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Types' is nil"))
		}
		if defsMap == nil {
			addAnalysisError(info, errors.New("type info map 'Defs' is nil"))
		}
		if usesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Uses' is nil"))
		}
	} else if pkg == nil {
		addAnalysisError(info, errors.New("cannot perform type analysis: target package is nil"))
	} else {
		addAnalysisError(info, errors.New("cannot perform type analysis: pkg.TypesInfo is nil"))
	}

	innermostNode := path[0]
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
		info.CompositeLit = compLit
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[compLit]; ok {
				info.CompositeLitType = tv.Type
				if info.CompositeLitType == nil {
					addAnalysisError(info, fmt.Errorf("composite literal type resolved to nil at %s", posStr(compLit.Pos())))
				}
			} else {
				addAnalysisError(info, fmt.Errorf("missing type info for composite literal at %s", posStr(compLit.Pos())))
			}
		} else if hasTypeInfo && typesMap == nil {
			addAnalysisError(info, errors.New("cannot analyze composite literal: Types map is nil"))
		}
		return
	}
	callExpr, callExprOk := path[0].(*ast.CallExpr)
	if !callExprOk && len(path) > 1 {
		callExpr, callExprOk = path[1].(*ast.CallExpr)
	}
	if callExprOk && cursorPos > callExpr.Lparen && cursorPos <= callExpr.Rparen {
		info.CallExpr = callExpr
		info.CallArgIndex = calculateArgIndex(callExpr.Args, cursorPos)
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[callExpr.Fun]; ok && tv.Type != nil {
				if sig, ok := tv.Type.Underlying().(*types.Signature); ok {
					info.CallExprFuncType = sig
					info.ExpectedArgType = determineExpectedArgType(sig, info.CallArgIndex)
				} else {
					addAnalysisError(info, fmt.Errorf("type of call func (%T) at %s is %T, not signature", callExpr.Fun, posStr(callExpr.Fun.Pos()), tv.Type))
				}
			} else {
				if ok && tv.Type == nil {
					addAnalysisError(info, fmt.Errorf("type info resolved to nil for call func (%T) at %s", callExpr.Fun, posStr(callExpr.Fun.Pos())))
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info for call func (%T) at %s", callExpr.Fun, posStr(callExpr.Fun.Pos())))
				}
			}
		} else if hasTypeInfo && typesMap == nil {
			addAnalysisError(info, errors.New("cannot analyze call expr: Types map is nil"))
		}
		return
	}
	for i := 0; i < len(path) && i < 2; i++ {
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok && cursorPos > selExpr.X.End() {
			info.SelectorExpr = selExpr
			if hasTypeInfo && typesMap != nil {
				if tv, ok := typesMap[selExpr.X]; ok {
					info.SelectorExprType = tv.Type
					if tv.Type == nil {
						addAnalysisError(info, fmt.Errorf("missing type for selector base expr (%T) starting at %s", selExpr.X, posStr(selExpr.X.Pos())))
					}
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info entry for selector base expr (%T) starting at %s", selExpr.X, posStr(selExpr.X.Pos())))
				}
			} else if hasTypeInfo && typesMap == nil {
				addAnalysisError(info, errors.New("cannot analyze selector expr: Types map is nil"))
			}
			return
		}
	}
	var ident *ast.Ident
	if id, ok := path[0].(*ast.Ident); ok && cursorPos == id.End() {
		ident = id
	} else if len(path) > 1 {
		if id, ok := path[1].(*ast.Ident); ok && cursorPos >= id.Pos() && cursorPos <= id.End() {
			if _, pIsSel := path[0].(*ast.SelectorExpr); !pIsSel || path[0].(*ast.SelectorExpr).Sel != id {
				ident = id
			}
		}
	}
	if ident != nil {
		info.IdentifierAtCursor = ident
		if hasTypeInfo {
			var obj types.Object
			if usesMap != nil {
				obj = usesMap[ident]
			}
			if obj == nil && defsMap != nil {
				obj = defsMap[ident]
			}
			if obj != nil {
				info.IdentifierObject = obj
				info.IdentifierType = obj.Type()
				if info.IdentifierType == nil {
					addAnalysisError(info, fmt.Errorf("object '%s' at %s found but type is nil", obj.Name(), posStr(obj.Pos())))
				}
			} else {
				if typesMap != nil {
					if tv, ok := typesMap[ident]; ok && tv.Type != nil {
						info.IdentifierType = tv.Type
					} else {
						if defsMap != nil && usesMap != nil {
							addAnalysisError(info, fmt.Errorf("missing object and type info for identifier '%s' at %s", ident.Name, posStr(ident.Pos())))
						}
					}
				} else if defsMap == nil && usesMap == nil {
					addAnalysisError(info, fmt.Errorf("missing object info for identifier '%s' at %s (defs/uses/types maps nil)", ident.Name, posStr(ident.Pos())))
				} else {
					addAnalysisError(info, fmt.Errorf("missing object info for identifier '%s' at %s", ident.Name, posStr(ident.Pos())))
				}
			}
		} else {
			addAnalysisError(info, errors.New("missing type info for identifier analysis"))
		}
	}
}

func calculateArgIndex(args []ast.Expr, cursorPos token.Pos) int {
	if len(args) == 0 {
		return 0
	}
	for i, arg := range args {
		if arg == nil {
			continue
		}
		argStart := arg.Pos()
		argEnd := arg.End()
		slotStart := argStart
		if i > 0 && args[i-1] != nil {
			slotStart = args[i-1].End() + 1
		}
		if cursorPos >= slotStart && cursorPos <= argEnd {
			return i
		}
		if cursorPos > arg.End() {
			if i == len(args)-1 {
				return i + 1
			}
		} else if cursorPos > arg.End() {
			return i + 1
		}
	}
	if len(args) > 0 && args[0] != nil && cursorPos < args[0].Pos() {
		return 0
	}
	if len(args) > 0 && args[len(args)-1] != nil && cursorPos > args[len(args)-1].End() {
		return len(args)
	}
	return 0
}

func determineExpectedArgType(sig *types.Signature, argIndex int) types.Type {
	if sig == nil || argIndex < 0 {
		return nil
	}
	params := sig.Params()
	if params == nil {
		return nil
	}
	numParams := params.Len()
	if numParams == 0 {
		return nil
	}
	if sig.Variadic() {
		if argIndex >= numParams-1 {
			lastParam := params.At(numParams - 1)
			if lastParam == nil {
				return nil
			}
			if slice, ok := lastParam.Type().(*types.Slice); ok {
				return slice.Elem()
			}
			return nil
		}
		param := params.At(argIndex)
		if param == nil {
			return nil
		}
		return param.Type()
	} else {
		if argIndex < numParams {
			param := params.At(argIndex)
			if param == nil {
				return nil
			}
			return param.Type()
		}
	}
	return nil
}

func listTypeMembers(typ types.Type, expr ast.Expr, qualifier types.Qualifier) []MemberInfo {
	if typ == nil {
		return nil
	}
	var members []MemberInfo
	currentType := typ
	isPointer := false
	if ptr, ok := typ.(*types.Pointer); ok {
		if ptr.Elem() == nil {
			return nil
		}
		currentType = ptr.Elem()
		isPointer = true
	}
	if currentType == nil {
		return nil
	}
	underlying := currentType.Underlying()
	if underlying == nil {
		return nil
	}

	switch u := underlying.(type) {
	case *types.Basic:
		members = append(members, MemberInfo{Name: u.String(), Kind: OtherMember, TypeString: "(basic type)"})
	case *types.Map:
		members = append(members, MemberInfo{Name: "// map operations", Kind: OtherMember, TypeString: "make, len, delete, range"})
	case *types.Slice:
		members = append(members, MemberInfo{Name: "// slice operations", Kind: OtherMember, TypeString: "make, len, cap, append, copy, range"})
	case *types.Chan:
		members = append(members, MemberInfo{Name: "// channel operations", Kind: OtherMember, TypeString: "make, len, cap, close, send (<-), receive (<-)"})
	case *types.Struct:
		members = append(members, listStructFields(u, qualifier)...)
	case *types.Interface:
		for i := 0; i < u.NumExplicitMethods(); i++ {
			method := u.ExplicitMethod(i)
			if method != nil && method.Exported() {
				members = append(members, MemberInfo{Name: method.Name(), Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
			}
		}
		for i := 0; i < u.NumEmbeddeds(); i++ {
			embeddedType := u.EmbeddedType(i)
			if embeddedType != nil {
				members = append(members, MemberInfo{Name: "// embeds", Kind: OtherMember, TypeString: types.TypeString(embeddedType, qualifier)})
			}
		}
	}
	mset := types.NewMethodSet(typ)
	for i := 0; i < mset.Len(); i++ {
		sel := mset.At(i)
		if sel != nil {
			methodObj := sel.Obj()
			if method, ok := methodObj.(*types.Func); ok {
				if method != nil && method.Exported() {
					members = append(members, MemberInfo{Name: method.Name(), Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
				}
			}
		}
	}
	if !isPointer {
		var ptrType types.Type
		if _, isNamed := currentType.(*types.Named); isNamed {
			ptrType = types.NewPointer(currentType)
		} else if _, isBasic := currentType.(*types.Basic); isBasic {
			ptrType = types.NewPointer(currentType)
		}
		if ptrType != nil {
			msetPtr := types.NewMethodSet(ptrType)
			existingMethods := make(map[string]struct{})
			for _, m := range members {
				if m.Kind == MethodMember {
					existingMethods[m.Name] = struct{}{}
				}
			}
			for i := 0; i < msetPtr.Len(); i++ {
				sel := msetPtr.At(i)
				if sel != nil {
					methodObj := sel.Obj()
					if method, ok := methodObj.(*types.Func); ok {
						if method != nil && method.Exported() {
							if _, exists := existingMethods[method.Name()]; !exists {
								members = append(members, MemberInfo{Name: method.Name(), Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
							}
						}
					}
				}
			}
		}
	}
	if len(members) > 0 {
		seen := make(map[string]struct{})
		uniqueMembers := make([]MemberInfo, 0, len(members))
		for _, m := range members {
			key := string(m.Kind) + ":" + m.Name
			if _, ok := seen[key]; !ok {
				seen[key] = struct{}{}
				uniqueMembers = append(uniqueMembers, m)
			}
		}
		members = uniqueMembers
	}
	return members
}

func listStructFields(st *types.Struct, qualifier types.Qualifier) []MemberInfo {
	var fields []MemberInfo
	if st == nil {
		return nil
	}
	for i := 0; i < st.NumFields(); i++ {
		field := st.Field(i)
		if field != nil && field.Exported() {
			fields = append(fields, MemberInfo{Name: field.Name(), Kind: FieldMember, TypeString: types.TypeString(field.Type(), qualifier)})
		}
	}
	return fields
}

func logAnalysisErrors(errs []error) {
	if len(errs) > 0 {
		combinedErr := errors.Join(errs...)
		log.Printf("Context analysis completed with %d non-fatal error(s): %v", len(errs), combinedErr)
	}
}

func addAnalysisError(info *AstContextInfo, err error) {
	if err != nil && info != nil {
		errMsg := err.Error()
		for _, existing := range info.AnalysisErrors {
			if existing.Error() == errMsg {
				return
			}
		}
		log.Printf("Analysis Warning: %v", err)
		info.AnalysisErrors = append(info.AnalysisErrors, err)
	}
}

// ============================================================================
// LSP Position Conversion Helpers
// ============================================================================
func LspPositionToBytePosition(content []byte, lspPos LSPPosition) (line, col, byteOffset int, err error) {
	if content == nil {
		return 0, 0, -1, fmt.Errorf("%w: file content is nil", ErrPositionConversion)
	}
	targetLine := int(lspPos.Line)
	targetUTF16Char := int(lspPos.Character)
	if targetLine < 0 {
		return 0, 0, -1, fmt.Errorf("%w: invalid LSP line: %d (must be >= 0)", ErrPositionConversion, targetLine)
	}
	if targetUTF16Char < 0 {
		return 0, 0, -1, fmt.Errorf("%w: invalid LSP character offset: %d (must be >= 0)", ErrPositionConversion, targetUTF16Char)
	}

	currentLine := 0
	currentByteOffset := 0
	scanner := bufio.NewScanner(bytes.NewReader(content))
	for scanner.Scan() {
		lineTextBytes := scanner.Bytes()
		lineLengthBytes := len(lineTextBytes)
		newlineLengthBytes := 1 // Assume \n
		if currentLine == targetLine {
			byteOffsetInLine, convErr := Utf16OffsetToBytes(lineTextBytes, targetUTF16Char)
			if convErr != nil {
				log.Printf("Warning: utf16OffsetToBytes failed (line %d, char %d): %v. Clamping to line end.", targetLine, targetUTF16Char, convErr)
				byteOffsetInLine = lineLengthBytes
			}
			line = currentLine + 1
			col = byteOffsetInLine + 1
			byteOffset = currentByteOffset + byteOffsetInLine
			return line, col, byteOffset, nil
		}
		currentByteOffset += lineLengthBytes + newlineLengthBytes
		currentLine++
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, -1, fmt.Errorf("%w: error scanning file content: %w", ErrPositionConversion, err)
	}
	if currentLine == targetLine { // Cursor on line after last line
		if targetUTF16Char == 0 {
			line = currentLine + 1
			col = 1
			byteOffset = currentByteOffset
			return line, col, byteOffset, nil
		} else {
			return 0, 0, -1, fmt.Errorf("%w: invalid character offset %d on line %d (after last line with content)", ErrPositionConversion, targetUTF16Char, targetLine)
		}
	}
	return 0, 0, -1, fmt.Errorf("%w: LSP line %d not found in file (total lines scanned %d)", ErrPositionConversion, targetLine, currentLine)
}

func Utf16OffsetToBytes(line []byte, utf16Offset int) (int, error) {
	if utf16Offset < 0 {
		return 0, fmt.Errorf("%w: invalid utf16Offset: %d (must be >= 0)", ErrPositionConversion, utf16Offset)
	}
	if utf16Offset == 0 {
		return 0, nil
	}
	byteOffset := 0
	currentUTF16Offset := 0
	for byteOffset < len(line) {
		if currentUTF16Offset >= utf16Offset {
			break
		}
		r, size := utf8.DecodeRune(line[byteOffset:])
		if r == utf8.RuneError && size <= 1 {
			return byteOffset, fmt.Errorf("%w: invalid UTF-8 sequence at byte offset %d", ErrPositionConversion, byteOffset)
		}
		utf16Units := 1
		if r > 0xFFFF {
			utf16Units = 2
		}
		if currentUTF16Offset+utf16Units > utf16Offset {
			break
		}
		currentUTF16Offset += utf16Units
		byteOffset += size
		if currentUTF16Offset == utf16Offset {
			break
		}
	}
	if currentUTF16Offset < utf16Offset {
		return len(line), fmt.Errorf("%w: utf16Offset %d is beyond the line length in UTF-16 units (%d)", ErrPositionConversion, utf16Offset, currentUTF16Offset)
	}
	return byteOffset, nil
}

// ============================================================================
// Cache Helper Functions (Restored)
// ============================================================================
func calculateGoModHash(dir string) string {
	goModPath := filepath.Join(dir, "go.mod")
	f, err := os.Open(goModPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "no-gomod"
		}
		log.Printf("Warning: Error reading %s: %v", goModPath, err)
		return "read-error"
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		log.Printf("Warning: Error hashing %s: %v", goModPath, err)
		return "hash-error"
	}
	return hex.EncodeToString(h.Sum(nil))
}

func calculateInputHashes(dir string, pkg *packages.Package) (map[string]string, error) {
	hashes := make(map[string]string)
	filesToHash := make(map[string]struct{})
	for _, fname := range []string{"go.mod", "go.sum"} {
		fpath := filepath.Join(dir, fname)
		if _, err := os.Stat(fpath); err == nil {
			absPath, absErr := filepath.Abs(fpath)
			if absErr == nil {
				filesToHash[absPath] = struct{}{}
			} else {
				log.Printf("Warning: Could not get absolute path for %s: %v", fpath, absErr)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("failed to stat %s: %w", fpath, err)
		}
	}
	filesFromPkg := false
	if pkg != nil && len(pkg.CompiledGoFiles) > 0 {
		filesFromPkg = true
		for _, fpath := range pkg.CompiledGoFiles {
			absPath, err := filepath.Abs(fpath)
			if err == nil {
				filesToHash[absPath] = struct{}{}
			} else {
				log.Printf("Warning: Could not get absolute path for %s: %v", fpath, err)
			}
		}
	}
	if !filesFromPkg {
		log.Printf("DEBUG: Calculating input hashes by scanning directory %s (pkg info unavailable)", dir)
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, fmt.Errorf("failed to scan directory %s for hashing: %w", dir, err)
		}
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") {
				absPath := filepath.Join(dir, entry.Name())
				absPath, absErr := filepath.Abs(absPath)
				if absErr == nil {
					filesToHash[absPath] = struct{}{}
				} else {
					log.Printf("Warning: Could not get absolute path for %s: %v", entry.Name(), absErr)
				}
			}
		}
	}
	for absPath := range filesToHash {
		relPath, err := filepath.Rel(dir, absPath)
		if err != nil {
			log.Printf("Warning: Could not get relative path for %s in %s: %v", absPath, dir, err)
			relPath = filepath.Base(absPath)
		}
		hash, err := hashFileContent(absPath)
		if err != nil {
			return nil, fmt.Errorf("failed to hash input file %s: %w", absPath, err)
		}
		hashes[relPath] = hash
	}
	return hashes, nil
}

func hashFileContent(filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func compareFileHashes(current, cached map[string]string) bool {
	if len(current) != len(cached) {
		log.Printf("DEBUG: Cache invalid: File count mismatch (%d vs %d cached)", len(current), len(cached))
		return false
	}
	for relPath, currentHash := range current {
		cachedHash, ok := cached[relPath]
		if !ok {
			log.Printf("DEBUG: Cache invalid: File %s missing in cache.", relPath)
			return false
		}
		if currentHash != cachedHash {
			log.Printf("DEBUG: Cache invalid: Hash mismatch for file %s.", relPath)
			return false
		}
	}
	for relPath := range cached {
		if _, ok := current[relPath]; !ok {
			log.Printf("DEBUG: Cache invalid: Cached file %s not found in current inputs.", relPath)
			return false
		}
	}
	return true
}

func (a *GoPackagesAnalyzer) deleteCacheEntry(cacheKey []byte) {
	if a.db == nil {
		return
	}
	err := a.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(cacheBucketName)
		if b == nil {
			return nil
		}
		if b.Get(cacheKey) == nil {
			return nil
		}
		log.Printf("DEBUG: Deleting cache entry for key: %s", string(cacheKey))
		return b.Delete(cacheKey)
	})
	if err != nil {
		log.Printf("Warning: Failed to delete cache entry %s: %v", string(cacheKey), err)
	}
}

// ============================================================================
// Spinner & File Helpers
// ============================================================================

// Spinner provides visual feedback for long-running operations.
type Spinner struct {
	chars    []string
	message  string // Current message to display
	index    int
	mu       sync.Mutex // Protects message and index
	stopChan chan struct{}
	doneChan chan struct{} // Used for graceful shutdown confirmation
	running  bool
}

// NewSpinner creates and initializes a new Spinner. // Function restored
func NewSpinner() *Spinner {
	return &Spinner{
		chars: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}, // Default spinner characters
		index: 0,
	}
}

// Start begins the spinner animation in a separate goroutine.
func (s *Spinner) Start(initialMessage string) {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return // Already running
	}
	s.stopChan = make(chan struct{})
	s.doneChan = make(chan struct{})
	s.message = initialMessage
	s.running = true
	s.mu.Unlock()

	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		defer func() {
			s.mu.Lock()
			isRunning := s.running
			s.running = false // Mark as not running *before* potentially clearing line
			s.mu.Unlock()
			if isRunning {
				fmt.Fprintf(os.Stderr, "\r\033[K") // Clear line on stop only if it was running
			}
			select {
			case s.doneChan <- struct{}{}: // Signal done
			default: // Avoid blocking if Stop already timed out
			}
			close(s.doneChan) // Close after signaling
		}()

		for {
			select {
			case <-s.stopChan:
				return // Exit goroutine when stopChan is closed
			case <-ticker.C:
				s.mu.Lock()
				if !s.running { // Check running flag inside loop as well
					s.mu.Unlock()
					return // Exit if stopped between ticks
				}
				char := s.chars[s.index]
				msg := s.message
				s.index = (s.index + 1) % len(s.chars)
				// Use stderr for spinner to avoid interfering with completion output on stdout
				fmt.Fprintf(os.Stderr, "\r\033[K%s%s%s %s", ColorCyan, char, ColorReset, msg)
				s.mu.Unlock()
			}
		}
	}()
}

// UpdateMessage changes the message displayed next to the spinner.
func (s *Spinner) UpdateMessage(newMessage string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running { // Only update if actually running
		s.message = newMessage
	}
}

// Stop halts the spinner animation and cleans up resources.
func (s *Spinner) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return // Already stopped
	}
	select {
	case <-s.stopChan: // Already closed? Should not happen if running=true
	default:
		close(s.stopChan) // Close channel to signal goroutine
	}
	doneChan := s.doneChan // Read under lock
	s.mu.Unlock()          // Unlock before potentially waiting

	if doneChan != nil {
		// Wait with timeout for the goroutine to finish cleanup
		select {
		case <-doneChan: // Wait for goroutine to signal done
		case <-time.After(500 * time.Millisecond):
			log.Println("Warning: Timeout waiting for spinner goroutine cleanup")
		}
	}
	// Ensure line is cleared finally, regardless of goroutine state after timeout
	fmt.Fprintf(os.Stderr, "\r\033[K")
}

// ============================================================================
// Snippet Extraction Helper (Restored)
// ============================================================================

// extractSnippetContext reads the file and extracts the code prefix/suffix around the cursor.
func extractSnippetContext(filename string, row, col int) (SnippetContext, error) {
	var ctx SnippetContext
	contentBytes, err := os.ReadFile(filename)
	if err != nil {
		return ctx, fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	content := string(contentBytes)
	fset := token.NewFileSet()
	file := fset.AddFile(filename, 1, len(contentBytes))
	if file == nil {
		return ctx, fmt.Errorf("failed to add file '%s' to fileset", filename)
	}

	cursorPos, posErr := calculateCursorPos(file, row, col)
	if posErr != nil {
		return ctx, fmt.Errorf("cannot determine valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		return ctx, fmt.Errorf("invalid cursor position calculated (Pos: %d)", cursorPos)
	}

	offset := file.Offset(cursorPos)
	if offset < 0 {
		offset = 0
	}
	if offset > len(content) {
		offset = len(content)
	}

	ctx.Prefix = content[:offset]
	ctx.Suffix = content[offset:]

	// Extract full line
	lineStartPos := file.LineStart(row)
	if !lineStartPos.IsValid() {
		log.Printf("Warning: Could not get start position for line %d in %s", row, filename)
		return ctx, nil
	}
	startOffset := file.Offset(lineStartPos)

	lineEndOffset := file.Size() // Default for last line
	fileLineCount := file.LineCount()
	if row < fileLineCount {
		nextLineStartPos := file.LineStart(row + 1)
		if nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	if startOffset >= 0 && lineEndOffset >= startOffset && lineEndOffset <= len(content) {
		lineContent := content[startOffset:lineEndOffset]
		lineContent = strings.TrimRight(lineContent, "\n")
		lineContent = strings.TrimRight(lineContent, "\r")
		ctx.FullLine = lineContent
	} else {
		log.Printf("Warning: Could not extract full line for row %d (start %d, end %d, content len %d) in %s",
			row, startOffset, lineEndOffset, len(content), filename)
	}
	return ctx, nil
}
