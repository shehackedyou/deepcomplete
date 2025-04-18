// Package deepcomplete provides core logic for local code completion using LLMs.
package deepcomplete

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"go.etcd.io/bbolt"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

// =============================================================================
// Constants & Core Types
// =============================================================================

const (
	defaultOllamaURL = "http://localhost:11434"
	defaultModel     = "deepseek-coder-r2"
	// Standard prompt template used for completions.
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
	// FIM prompt template used for fill-in-the-middle tasks.
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

	maxRetries            = 3               // Default number of retries for Ollama API calls.
	retryDelay            = 2 * time.Second // Default delay between retries.
	defaultMaxTokens      = 256             // Default maximum tokens for LLM response.
	DefaultStop           = "\n"            // Default stop sequence for LLM. Exported for CLI use.
	defaultTemperature    = 0.1             // Default sampling temperature for LLM.
	defaultConfigFileName = "config.json"   // Default config file name.
	configDirName         = "deepcomplete"  // Subdirectory name for config/data.
	cacheSchemaVersion    = 2               // Used to invalidate cache if internal formats change.
)

var (
	cacheBucketName = []byte("AnalysisCache") // Name of the bbolt bucket for caching.
)

// Config holds the active configuration for the autocompletion service.
type Config struct {
	OllamaURL      string   `json:"ollama_url"`
	Model          string   `json:"model"`
	PromptTemplate string   `json:"-"` // Loaded internally, not from config file.
	FimTemplate    string   `json:"-"` // Loaded internally, not from config file.
	MaxTokens      int      `json:"max_tokens"`
	Stop           []string `json:"stop"`
	Temperature    float64  `json:"temperature"`
	UseAst         bool     `json:"use_ast"`          // Enable AST/Type analysis.
	UseFim         bool     `json:"use_fim"`          // Use Fill-in-the-Middle prompting.
	MaxPreambleLen int      `json:"max_preamble_len"` // Max bytes for AST context preamble.
	MaxSnippetLen  int      `json:"max_snippet_len"`  // Max bytes for code snippet context.
}

// Validate checks if configuration values are valid, applying defaults for some fields.
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
	// Apply defaults for invalid numeric values.
	if c.MaxTokens <= 0 {
		log.Printf("Warning: max_tokens (%d) is not positive, using default %d", c.MaxTokens, defaultMaxTokens)
		c.MaxTokens = defaultMaxTokens
	}
	if c.Temperature < 0 {
		log.Printf("Warning: temperature (%.2f) is negative, using default %.2f", c.Temperature, defaultTemperature)
		c.Temperature = defaultTemperature
	}
	if c.MaxPreambleLen <= 0 {
		log.Printf("Warning: max_preamble_len (%d) is not positive, using default %d", c.MaxPreambleLen, DefaultConfig.MaxPreambleLen)
		c.MaxPreambleLen = DefaultConfig.MaxPreambleLen
	}
	if c.MaxSnippetLen <= 0 {
		log.Printf("Warning: max_snippet_len (%d) is not positive, using default %d", c.MaxSnippetLen, DefaultConfig.MaxSnippetLen)
		c.MaxSnippetLen = DefaultConfig.MaxSnippetLen
	}
	return nil
}

// FileConfig represents the structure of the JSON config file for unmarshalling.
// Pointers distinguish between absent fields and zero values.
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

// AstContextInfo holds structured information extracted from code analysis.
// This is used internally to build the prompt preamble.
type AstContextInfo struct {
	FilePath           string
	CursorPos          token.Pos
	PackageName        string
	EnclosingFunc      *types.Func   // From type analysis.
	EnclosingFuncNode  *ast.FuncDecl // From AST.
	ReceiverType       string        // Formatted receiver if method.
	EnclosingBlock     *ast.BlockStmt
	Imports            []*ast.ImportSpec
	CommentsNearCursor []string
	IdentifierAtCursor *ast.Ident
	IdentifierType     types.Type
	IdentifierObject   types.Object
	SelectorExpr       *ast.SelectorExpr // e.g., x.Y
	SelectorExprType   types.Type        // Type of x.
	CallExpr           *ast.CallExpr     // e.g., fn(a, b)
	CallExprFuncType   *types.Signature  // Signature of fn.
	CallArgIndex       int               // 0-based index of cursor arg.
	ExpectedArgType    types.Type        // Expected type of arg at cursor.
	CompositeLit       *ast.CompositeLit // e.g., T{...}
	CompositeLitType   types.Type        // Type of T.
	VariablesInScope   map[string]types.Object
	PromptPreamble     string  // Generated context string (potentially from cache).
	AnalysisErrors     []error // Non-fatal errors encountered during analysis.
}

// SnippetContext holds the code prefix and suffix relative to the cursor.
type SnippetContext struct {
	Prefix   string // Code before cursor.
	Suffix   string // Code after cursor.
	FullLine string // Full line where cursor is located.
}

// OllamaError defines a custom error for Ollama API issues.
type OllamaError struct {
	Message string
	Status  int // HTTP status code, if available.
}

func (e *OllamaError) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("Ollama error: %s (Status: %d)", e.Message, e.Status)
	}
	return fmt.Sprintf("Ollama error: %s", e.Message)
}

// OllamaResponse represents the streaming response structure from Ollama.
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"`
}

// CachedAnalysisData holds derived information stored in the cache (gob-encoded).
type CachedAnalysisData struct {
	PackageName    string
	PromptPreamble string
}

// CachedAnalysisEntry represents the full structure stored in bbolt.
type CachedAnalysisEntry struct {
	SchemaVersion   int               // Cache format version.
	GoModHash       string            // Hash of go.mod.
	InputFileHashes map[string]string // Hashes of source files (key: relative path).
	AnalysisGob     []byte            // Gob-encoded CachedAnalysisData.
}

// LSPPosition represents a 0-based line/character offset (UTF-16).
type LSPPosition struct {
	Line      uint32
	Character uint32
}

// MemberKind defines the type of member (field or method).
type MemberKind string

const (
	FieldMember  MemberKind = "field"
	MethodMember MemberKind = "method"
	OtherMember  MemberKind = "other" // Built-ins, etc.
)

// MemberInfo holds structured information about a type member.
type MemberInfo struct {
	Name       string
	Kind       MemberKind
	TypeString string // Type representation.
}

// =============================================================================
// Exported Errors
// =============================================================================

var (
	ErrAnalysisFailed       = errors.New("code analysis failed")        // Non-fatal if preamble generated.
	ErrOllamaUnavailable    = errors.New("ollama API unavailable")      // Network or server error.
	ErrStreamProcessing     = errors.New("error processing LLM stream") // Issue reading/parsing stream.
	ErrConfig               = errors.New("configuration error")         // Issue loading/saving config.
	ErrInvalidConfig        = errors.New("invalid configuration")       // Provided config values are bad.
	ErrCache                = errors.New("cache operation failed")      // General cache error.
	ErrCacheRead            = errors.New("cache read failed")
	ErrCacheWrite           = errors.New("cache write failed")
	ErrCacheDecode          = errors.New("cache decode failed")
	ErrCacheEncode          = errors.New("cache encode failed")
	ErrCacheHash            = errors.New("cache hash calculation failed")
	ErrPositionConversion   = errors.New("position conversion failed") // LSP <-> byte offset.
	ErrInvalidPositionInput = errors.New("invalid input position")     // Line/col < 0, etc.
	ErrPositionOutOfRange   = errors.New("position out of range")      // Position outside document bounds.
	ErrInvalidUTF8          = errors.New("invalid utf-8 sequence")     // During position conversion.
)

// =============================================================================
// Interfaces for Components
// =============================================================================

// LLMClient defines interaction with the LLM API.
type LLMClient interface {
	GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error)
}

// Analyzer defines code context analysis.
type Analyzer interface {
	Analyze(ctx context.Context, filename string, line, col int) (*AstContextInfo, error)
	Close() error
	InvalidateCache(dir string) error
}

// PromptFormatter defines prompt construction.
type PromptFormatter interface {
	FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string
}

// =============================================================================
// Variables & Default Config
// =============================================================================

var (
	// DefaultConfig provides default settings.
	DefaultConfig = Config{
		OllamaURL:      defaultOllamaURL,
		Model:          defaultModel,
		PromptTemplate: promptTemplate,
		FimTemplate:    fimPromptTemplate,
		MaxTokens:      defaultMaxTokens,
		Stop:           []string{DefaultStop, "}", "//", "/*"}, // Common Go stop sequences.
		Temperature:    defaultTemperature,
		UseAst:         true,
		UseFim:         false,
		MaxPreambleLen: 2048,
		MaxSnippetLen:  2048,
	}

	// Terminal color codes.
	ColorReset  = "\033[0m"
	ColorGreen  = "\033[38;5;119m"
	ColorYellow = "\033[38;5;220m"
	ColorBlue   = "\033[38;5;153m"
	ColorRed    = "\033[38;5;203m"
	ColorCyan   = "\033[38;5;141m"
)

// =============================================================================
// Exported Helper Functions
// =============================================================================

// PrettyPrint prints colored text to stderr.
func PrettyPrint(color, text string) {
	fmt.Fprint(os.Stderr, color, text, ColorReset)
}

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadConfig loads configuration from standard locations, merges with defaults,
// and attempts to write a default config if none exists or is invalid.
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

	// Try loading from primary path first.
	if primaryPath != "" {
		loaded, loadErr := loadAndMergeConfig(primaryPath, &cfg)
		if loadErr != nil {
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr // Track parsing error specifically.
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", primaryPath, loadErr))
		}
		loadedFromFile = loaded
		if loadedFromFile && loadErr == nil {
			log.Printf("Loaded config from %s", primaryPath)
		} else if loadedFromFile {
			log.Printf("Attempted load from %s but failed.", primaryPath)
		}
	}

	// Try secondary path ONLY if primary was not found OR failed reading/parsing.
	primaryNotFound := len(loadErrors) > 0 && errors.Is(loadErrors[len(loadErrors)-1], os.ErrNotExist)
	if !loadedFromFile && secondaryPath != "" && (primaryNotFound || primaryPath == "") {
		loaded, loadErr := loadAndMergeConfig(secondaryPath, &cfg)
		if loadErr != nil {
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				if configParseError == nil { // Only store first parse error.
					configParseError = loadErr
				}
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", secondaryPath, loadErr))
		}
		loadedFromFile = loaded
		if loadedFromFile && loadErr == nil {
			log.Printf("Loaded config from %s", secondaryPath)
		} else if loadedFromFile {
			log.Printf("Attempted load from %s but failed.", secondaryPath)
		}
	}

	// Write default config if no file existed OR if parsing failed.
	loadSucceeded := loadedFromFile && configParseError == nil
	if !loadSucceeded {
		if configParseError != nil {
			log.Printf("Existing config file failed to parse: %v. Attempting to write default.", configParseError)
		} else {
			log.Println("No config file found. Attempting to write default.")
		}
		writePath := primaryPath // Prefer primary path for writing default.
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
		cfg = DefaultConfig // Use defaults if load failed or parse error occurred.
	}

	// Assign internal templates (not loaded from file).
	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	// Validate the final merged/default configuration.
	finalCfg := cfg
	if err := finalCfg.Validate(); err != nil {
		log.Printf("Warning: Config after load/merge failed validation: %v. Returning defaults.", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		// Ensure default config itself is valid.
		if valErr := DefaultConfig.Validate(); valErr != nil {
			// This is a critical internal error.
			return DefaultConfig, fmt.Errorf("default config is invalid: %w", valErr)
		}
		finalCfg = DefaultConfig // Fallback to pure defaults on validation failure.
	}

	// Return validated config and any non-fatal load errors wrapped in ErrConfig.
	if len(loadErrors) > 0 {
		// Use errors.Join to combine multiple load/write errors.
		return finalCfg, fmt.Errorf("%w: %w", ErrConfig, errors.Join(loadErrors...))
	}
	return finalCfg, nil
}

// getConfigPaths determines the primary (XDG) and secondary (~/.config) config paths.
func getConfigPaths() (primary string, secondary string, err error) {
	var cfgErr, homeErr error
	userConfigDir, cfgErr := os.UserConfigDir() // e.g., ~/.config or AppData
	if cfgErr == nil {
		primary = filepath.Join(userConfigDir, configDirName, defaultConfigFileName)
	} else {
		log.Printf("Warning: Could not determine user config directory: %v", cfgErr)
	}

	// Fallback using home directory if XDG path failed.
	homeDir, homeErr := os.UserHomeDir()
	if homeErr == nil {
		// Use ~/.config as secondary/fallback if XDG failed.
		secondary = filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
		if primary == "" && cfgErr != nil {
			primary = secondary // Promote secondary to primary if XDG failed.
			log.Printf("Using fallback primary config path: %s", primary)
			secondary = "" // Avoid checking ~/.config twice.
		}
		// Avoid setting secondary if it's the same as primary.
		if primary == secondary {
			secondary = ""
		}
	} else {
		log.Printf("Warning: Could not determine user home directory: %v", homeErr)
	}

	// Report error only if BOTH determination methods failed.
	if primary == "" && secondary == "" {
		err = fmt.Errorf("cannot determine config/home directories: config error: %v; home error: %v", cfgErr, homeErr)
	}
	return primary, secondary, err
}

// loadAndMergeConfig attempts to load config from a path and merge into cfg.
func loadAndMergeConfig(path string, cfg *Config) (loaded bool, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil // File not found is not an error here.
		}
		return true, fmt.Errorf("reading config file %q failed: %w", path, err) // Return true because file exists but read failed.
	}
	if len(data) == 0 {
		log.Printf("Warning: Config file exists but is empty: %s", path)
		return true, nil // File exists but is empty.
	}

	var fileCfg FileConfig
	if err := json.Unmarshal(data, &fileCfg); err != nil {
		return true, fmt.Errorf("parsing config file JSON %q failed: %w", path, err) // File exists but parse failed.
	}

	// Merge non-nil fields from fileCfg into cfg.
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

	return true, nil // Loaded and merged successfully.
}

// writeDefaultConfig creates the directory and writes the default config as JSON.
func writeDefaultConfig(path string, defaultConfig Config) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0750); err != nil { // Use appropriate permissions.
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}

	// Use a temporary struct to marshal only the exportable fields.
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
	// Write with restricted permissions.
	if err := os.WriteFile(path, jsonData, 0640); err != nil {
		return fmt.Errorf("failed to write default config file %s: %w", path, err)
	}
	log.Printf("Wrote default configuration to %s", path)
	return nil
}

// =============================================================================
// Default Component Implementations
// =============================================================================

// httpOllamaClient implements LLMClient using HTTP requests to Ollama.
type httpOllamaClient struct {
	httpClient *http.Client
}

func newHttpOllamaClient() *httpOllamaClient {
	// Configure HTTP client with a reasonable timeout.
	return &httpOllamaClient{httpClient: &http.Client{Timeout: 90 * time.Second}}
}

// GenerateStream sends a request to Ollama's /api/generate endpoint.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	base := strings.TrimSuffix(config.OllamaURL, "/")
	endpointURL := base + "/api/generate"

	u, err := url.Parse(endpointURL)
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL '%s': %w", endpointURL, err)
	}

	// Construct the JSON payload for Ollama API.
	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true, // Request streaming response.
		"options": map[string]interface{}{
			"temperature": config.Temperature,
			"num_ctx":     4096, // Consider making configurable?
			"top_p":       0.9,  // Consider making configurable?
			"stop":        config.Stop,
			"num_predict": config.MaxTokens,
		},
	}
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON payload: %w", err)
	}

	// Create and send the HTTP request.
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/x-ndjson") // Expect newline-delimited JSON.

	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Handle common network errors.
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		var netErr *net.OpError
		if errors.As(err, &netErr) && netErr.Op == "dial" {
			return nil, fmt.Errorf("%w: connection refused or network error connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		return nil, fmt.Errorf("%w: http request failed: %w", ErrOllamaUnavailable, err)
	}

	// Check for non-200 status codes.
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(failed to read error response body)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			// Try to extract specific error message from Ollama JSON response.
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error
			}
		}
		err = &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, err) // Wrap the specific OllamaError.
	}

	// Return the response body for streaming.
	return resp.Body, nil
}

// GoPackagesAnalyzer implements Analyzer using go/packages and bbolt caching.
type GoPackagesAnalyzer struct {
	db *bbolt.DB  // BoltDB handle for caching.
	mu sync.Mutex // Protects access to db handle during Close.
}

// NewGoPackagesAnalyzer initializes the analyzer and opens the bbolt cache DB.
func NewGoPackagesAnalyzer() *GoPackagesAnalyzer {
	dbPath := ""
	// Determine cache directory path.
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		// Include schema version in path to auto-invalidate old caches.
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
		// Open bbolt database with a timeout.
		opts := &bbolt.Options{Timeout: 1 * time.Second}
		db, err = bbolt.Open(dbPath, 0600, opts) // Restrictive file permissions.
		if err != nil {
			log.Printf("Warning: Failed to open bbolt cache file %s: %v. Caching will be disabled.", dbPath, err)
			db = nil // Ensure db is nil if open fails.
		} else {
			// Ensure the cache bucket exists.
			err = db.Update(func(tx *bbolt.Tx) error {
				_, err := tx.CreateBucketIfNotExists(cacheBucketName)
				if err != nil {
					return fmt.Errorf("failed to create cache bucket %s: %w", string(cacheBucketName), err)
				}
				return nil
			})
			if err != nil {
				log.Printf("Warning: Failed to ensure bbolt bucket exists: %v. Caching disabled.", err)
				db.Close() // Close DB if bucket creation failed.
				db = nil
			} else {
				log.Printf("Using bbolt cache at %s (Schema v%d)", dbPath, cacheSchemaVersion)
			}
		}
	}
	return &GoPackagesAnalyzer{db: db}
}

// Close closes the bbolt database connection if open.
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock() // Ensure exclusive access to db handle during close.
	defer a.mu.Unlock()
	if a.db != nil {
		log.Println("Closing bbolt cache database.")
		err := a.db.Close()
		a.db = nil // Set to nil after closing.
		return err
	}
	return nil
}

// Analyze performs code analysis, utilizing the cache if possible.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, filename string, line, col int) (info *AstContextInfo, analysisErr error) {
	info = &AstContextInfo{
		FilePath:         filename,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		CallArgIndex:     -1, // Default value.
	}
	// Recover from potential panics during analysis.
	defer func() {
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			addAnalysisError(info, panicErr) // Log panic as analysis error.
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr) // Append panic error.
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

	// --- Cache Check ---
	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash) // Cache key based on dir and go.mod hash.
	var loadDuration, stepsDuration, preambleDuration time.Duration
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry

	// Try reading from cache if db is available.
	if a.db != nil {
		readStart := time.Now()
		dbErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				// Bucket might disappear if DB file is corrupted/deleted externally.
				return fmt.Errorf("%w: cache bucket %s not found during View", ErrCacheRead, string(cacheBucketName))
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				return nil // Cache miss.
			}

			// Decode the cached entry header.
			var decoded CachedAnalysisEntry
			decoder := gob.NewDecoder(bytes.NewReader(valBytes))
			if err := decoder.Decode(&decoded); err != nil {
				log.Printf("Warning: Failed to gob-decode cached entry header for key %s: %v. Treating as miss.", string(cacheKey), err)
				// Return wrapped error to indicate decode failure.
				return fmt.Errorf("%w: %w", ErrCacheDecode, err)
			}
			// Check schema version for compatibility.
			if decoded.SchemaVersion != cacheSchemaVersion {
				log.Printf("Warning: Cache data for key %s has old schema version %d (want %d). Ignoring.", string(cacheKey), decoded.SchemaVersion, cacheSchemaVersion)
				return nil // Treat as miss if schema differs.
			}
			cachedEntry = &decoded // Store decoded entry for hash validation.
			return nil
		})
		// Handle cache read/decode errors.
		if dbErr != nil {
			log.Printf("Warning: Error reading or decoding from bbolt cache: %v", dbErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbErr)) // Log as non-fatal analysis error.
			// If decode failed, try to delete the corrupted entry.
			if errors.Is(dbErr, ErrCacheDecode) {
				a.deleteCacheEntry(cacheKey)
			}
		}
		log.Printf("DEBUG: Cache read attempt took %v", time.Since(readStart))
	} else {
		log.Printf("DEBUG: Cache disabled (db handle is nil).")
	}

	// Validate cache hit based on file hashes.
	if cachedEntry != nil {
		validationStart := time.Now()
		log.Printf("DEBUG: Potential cache hit for key: %s. Validating file hashes...", string(cacheKey))
		// Calculate current hashes (pass nil pkg, load only on miss).
		currentHashes, hashErr := calculateInputHashes(dir, nil)
		// Check go.mod hash and individual file hashes.
		if hashErr == nil &&
			cachedEntry.GoModHash == goModHash &&
			compareFileHashes(currentHashes, cachedEntry.InputFileHashes) { // compareFileHashes logs details on mismatch.

			log.Printf("DEBUG: Cache VALID for key: %s. Attempting to decode analysis data...", string(cacheKey))
			decodeStart := time.Now()
			// Decode the actual analysis data (preamble).
			var analysisData CachedAnalysisData
			decoder := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob))
			if decodeErr := decoder.Decode(&analysisData); decodeErr == nil {
				// Cache hit and decode successful! Use cached data.
				info.PackageName = analysisData.PackageName
				info.PromptPreamble = analysisData.PromptPreamble
				cacheHit = true
				loadDuration = time.Since(decodeStart) // Use decode time as load time for hits.
				log.Printf("DEBUG: Analysis data successfully decoded from cache in %v.", loadDuration)
				log.Printf("DEBUG: Using cached preamble (length %d). Skipping packages.Load and analysis steps.", len(info.PromptPreamble))
			} else {
				// Failed to decode analysis data - cache entry is corrupt.
				log.Printf("Warning: Failed to gob-decode cached analysis data: %v. Treating as miss.", decodeErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheDecode, decodeErr))
				a.deleteCacheEntry(cacheKey) // Delete corrupt entry.
			}
		} else {
			// Cache is invalid (hash mismatch or error calculating current hashes).
			log.Printf("DEBUG: Cache INVALID for key: %s (HashErr: %v). Treating as miss.", string(cacheKey), hashErr)
			a.deleteCacheEntry(cacheKey) // Delete invalid entry.
			if hashErr != nil {
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr))
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

		// 1. Load Go package information.
		loadStart := time.Now()
		fset := token.NewFileSet()
		targetPkg, targetFileAST, targetFile, loadErrors := loadPackageInfo(ctx, absFilename, fset)
		loadDuration = time.Since(loadStart)
		log.Printf("DEBUG: packages.Load completed in %v", loadDuration)
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr) // Add errors from package loading.
		}

		// 2. Perform detailed AST/Type analysis steps if loading was successful enough.
		if targetFile != nil && fset != nil {
			stepsStart := time.Now()
			analyzeStepErr := a.performAnalysisSteps(targetFile, targetFileAST, targetPkg, fset, line, col, info)
			stepsDuration = time.Since(stepsStart)
			log.Printf("DEBUG: performAnalysisSteps completed in %v", stepsDuration)
			if analyzeStepErr != nil {
				addAnalysisError(info, analyzeStepErr)
			}
		} else {
			// Log if analysis cannot proceed due to load failures.
			if targetFile == nil {
				addAnalysisError(info, errors.New("cannot perform analysis steps: missing target file after load"))
			}
			if fset == nil && targetFile != nil {
				addAnalysisError(info, errors.New("cannot perform analysis steps: missing FileSet after load"))
			}
		}

		// 3. Build the context preamble string from analysis results.
		var qualifier types.Qualifier // Used for formatting type names relative to the package.
		if targetPkg != nil && targetPkg.Types != nil {
			info.PackageName = targetPkg.Types.Name()
			qualifier = types.RelativeTo(targetPkg.Types)
		} else {
			// Fallback qualifier if type info is missing.
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

		// 4. Save results to cache if enabled, preamble generated, and no load errors.
		// Only cache results from successful loads to avoid caching partial/incorrect data.
		if a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0 {
			log.Printf("DEBUG: Attempting to save analysis results to bbolt cache. Key: %s", string(cacheKey))
			saveStart := time.Now()
			// Use the loaded package info for hashing now.
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg)
			if hashErr == nil {
				// Prepare data for gob encoding.
				analysisDataToCache := CachedAnalysisData{
					PackageName:    info.PackageName,
					PromptPreamble: info.PromptPreamble,
				}
				var gobBuf bytes.Buffer
				encoder := gob.NewEncoder(&gobBuf)
				if encodeErr := encoder.Encode(&analysisDataToCache); encodeErr == nil {
					analysisGob := gobBuf.Bytes()
					// Prepare the full cache entry with metadata.
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
						// Write to bbolt database.
						saveErr := a.db.Update(func(tx *bbolt.Tx) error {
							b := tx.Bucket(cacheBucketName)
							if b == nil {
								// Should not happen if initialization succeeded.
								return fmt.Errorf("%w: cache bucket %s disappeared during save", ErrCacheWrite, string(cacheBucketName))
							}
							log.Printf("DEBUG: Writing %d bytes to cache for key %s", len(encodedBytes), string(cacheKey))
							return b.Put(cacheKey, encodedBytes)
						})
						if saveErr == nil {
							log.Printf("DEBUG: Saved analysis results to bbolt cache %s in %v", string(cacheKey), time.Since(saveStart))
						} else {
							log.Printf("Warning: Failed to write to bbolt cache for key %s: %v", string(cacheKey), saveErr)
							addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheWrite, saveErr))
						}
					} else {
						log.Printf("Warning: Failed to gob-encode cache entry for key %s: %v", string(cacheKey), entryEncodeErr)
						addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr))
					}
				} else {
					log.Printf("Warning: Failed to gob-encode analysis data for caching: %v", encodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr))
				}
			} else {
				log.Printf("Warning: Failed to calculate input hashes for caching: %v", hashErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr))
			}
		} else if a.db != nil {
			// Log why caching was skipped.
			log.Printf("DEBUG: Skipping cache save for key %s (Load Errors: %d, Preamble Empty: %t)",
				string(cacheKey), len(loadErrors), info.PromptPreamble == "")
		}
	}

	// Log final state and errors.
	logAnalysisErrors(info.AnalysisErrors)
	analysisErr = errors.Join(info.AnalysisErrors...) // Combine all non-fatal errors.

	// Log summary timing.
	if cacheHit {
		log.Printf("Context analysis finished (Decode: %v, Cache Hit: %t)", loadDuration, cacheHit)
	} else {
		log.Printf("Context analysis finished (Load: %v, Steps: %v, Preamble: %v, Cache Hit: %t)",
			loadDuration, stepsDuration, preambleDuration, cacheHit)
	}
	// Log truncated preamble for context.
	if len(info.PromptPreamble) > 500 {
		log.Printf("Final Context Preamble (length %d):\n---\n%s\n... (preamble truncated in log)\n---", len(info.PromptPreamble), info.PromptPreamble[:500])
	} else {
		log.Printf("Final Context Preamble (length %d):\n---\n%s\n---", len(info.PromptPreamble), info.PromptPreamble)
	}

	// Determine final return error: Wrap combined non-fatal errors in ErrAnalysisFailed.
	if analysisErr != nil {
		// Return non-fatal error even if preamble was generated.
		return info, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisErr)
	}
	return info, nil // Success.
}

// InvalidateCache removes the cached entry for a given directory.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	a.mu.Lock() // Lock needed only if db handle itself could change, which it doesn't here. Minimal impact.
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
			return nil // Not an error if bucket doesn't exist.
		}
		// Check if key exists before attempting delete.
		if b.Get(cacheKey) == nil {
			log.Printf("DEBUG: Cache entry for key %s already deleted or never existed.", string(cacheKey))
			return nil // Not an error if key doesn't exist.
		}
		log.Printf("DEBUG: Deleting cache entry for key: %s", string(cacheKey))
		return b.Delete(cacheKey)
	})
	if err != nil {
		log.Printf("Warning: Failed to delete cache entry %s: %v", string(cacheKey), err)
		// Wrap error for clarity.
		return fmt.Errorf("%w: failed to delete entry %s: %w", ErrCacheWrite, string(cacheKey), err)
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

	// Calculate cursor position offset.
	cursorPos, posErr := calculateCursorPos(targetFile, line, col)
	if posErr != nil {
		// calculateCursorPos now returns wrapped errors.
		return fmt.Errorf("%w: %w", ErrPositionConversion, posErr)
	}
	info.CursorPos = cursorPos
	log.Printf("Calculated cursor token.Pos: %d (%s)", info.CursorPos, fset.PositionFor(info.CursorPos, true))

	// Perform AST-based analysis if AST is available.
	if targetFileAST != nil {
		path := findEnclosingPath(targetFileAST, info.CursorPos, info)
		findContextNodes(path, info.CursorPos, targetPkg, fset, info)         // Find nodes like CallExpr, SelectorExpr etc.
		gatherScopeContext(path, targetPkg, fset, info)                       // Find enclosing func, block, vars in scope.
		findRelevantComments(targetFileAST, path, info.CursorPos, fset, info) // Find associated comments.
	} else {
		addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil"))
		// Still try to get package scope even without specific file AST.
		gatherScopeContext(nil, targetPkg, fset, info)
	}
	// Consolidate errors gathered during steps? Currently handled by returning joined info.AnalysisErrors.
	return nil
}

// --- Prompt Formatter ---
type templateFormatter struct{}

func newTemplateFormatter() *templateFormatter { return &templateFormatter{} }

// FormatPrompt combines context and snippet into the final LLM prompt, applying truncation.
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate
	// Use limits from config.
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2 // Split snippet budget for FIM.

	// Truncate preamble (keeping the end, as it often contains more specific context).
	if len(contextPreamble) > maxPreambleLen {
		log.Printf("Warning: Truncating context preamble from %d to %d bytes.", len(contextPreamble), maxPreambleLen)
		// Calculate start index to keep the last maxPreambleLen bytes, accounting for marker length.
		marker := "... (context truncated)\n"
		startByte := len(contextPreamble) - maxPreambleLen + len(marker)
		if startByte < 0 {
			startByte = 0
		} // Ensure start index is not negative.
		contextPreamble = marker + contextPreamble[startByte:]
	}

	if config.UseFim { // Format for Fill-in-the-Middle.
		template = config.FimTemplate
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix

		// Truncate FIM prefix (keep end).
		if len(prefix) > maxFIMPartLen {
			log.Printf("Warning: Truncating FIM prefix from %d to %d bytes.", len(prefix), maxFIMPartLen)
			marker := "...(prefix truncated)"
			startByte := len(prefix) - maxFIMPartLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			prefix = marker + prefix[startByte:]
		}
		// Truncate FIM suffix (keep start).
		if len(suffix) > maxFIMPartLen {
			log.Printf("Warning: Truncating FIM suffix from %d to %d bytes.", len(suffix), maxFIMPartLen)
			marker := "(suffix truncated)..."
			endByte := maxFIMPartLen - len(marker)
			if endByte < 0 {
				endByte = 0
			}
			suffix = suffix[:endByte] + marker
		}
		finalPrompt = fmt.Sprintf(template, contextPreamble, prefix, suffix)

	} else { // Format for standard completion.
		snippet := snippetCtx.Prefix // Use only code before cursor.
		// Truncate snippet (keep end).
		if len(snippet) > maxSnippetLen {
			log.Printf("Warning: Truncating code snippet from %d to %d bytes.", len(snippet), maxSnippetLen)
			marker := "...(code truncated)\n"
			startByte := len(snippet) - maxSnippetLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			snippet = marker + snippet[startByte:]
		}
		finalPrompt = fmt.Sprintf(template, contextPreamble, snippet)
	}
	return finalPrompt
}

// =============================================================================
// DeepCompleter Service
// =============================================================================

// DeepCompleter orchestrates analysis, formatting, and LLM interaction.
type DeepCompleter struct {
	client    LLMClient
	analyzer  Analyzer
	formatter PromptFormatter
	config    Config
	configMu  sync.RWMutex // Protects concurrent read/write access to config.
}

// NewDeepCompleter creates a new DeepCompleter service with default config.
func NewDeepCompleter() (*DeepCompleter, error) {
	cfg, configErr := LoadConfig() // Load initial config.
	// Handle fatal vs non-fatal config load errors.
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		return nil, configErr // Fatal error during initial load.
	}
	if configErr != nil {
		log.Printf("Warning during initial config load: %v", configErr) // Log non-fatal errors.
	}

	// Ensure loaded config is valid, fallback to pure defaults if necessary.
	if err := cfg.Validate(); err != nil {
		log.Printf("Warning: Initial config is invalid after load/merge: %v. Using pure defaults.", err)
		cfg = DefaultConfig
		// This should not fail, but check defensively.
		if valErr := cfg.Validate(); valErr != nil {
			return nil, fmt.Errorf("default config validation failed: %w", valErr)
		}
	}

	analyzer := NewGoPackagesAnalyzer() // Initialize analyzer (handles its own cache setup).
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    cfg,
		// configMu is initialized automatically.
	}
	// Return the non-fatal config load error if one occurred.
	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with provided config.
func NewDeepCompleterWithConfig(config Config) (*DeepCompleter, error) {
	// Ensure internal templates are set if not provided.
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	// Validate the provided config.
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

// Close cleans up resources, primarily the analyzer's cache DB.
func (dc *DeepCompleter) Close() error {
	if dc.analyzer != nil {
		return dc.analyzer.Close()
	}
	return nil
}

// UpdateConfig atomically updates the completer's configuration after validation.
func (dc *DeepCompleter) UpdateConfig(newConfig Config) error {
	// Ensure internal templates are preserved/set.
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate
	}

	// Validate the incoming configuration.
	if err := newConfig.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	// Acquire write lock to update config safely.
	dc.configMu.Lock()
	defer dc.configMu.Unlock()
	dc.config = newConfig
	log.Printf("DeepCompleter configuration updated: %+v", newConfig)
	return nil
}

// GetCompletion provides basic completion for a direct code snippet (no AST analysis).
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	log.Println("DeepCompleter.GetCompletion called for basic prompt.")
	// Use read lock to access config safely.
	dc.configMu.RLock()
	currentConfig := dc.config // Make a copy while holding lock.
	dc.configMu.RUnlock()

	// Basic preamble for non-AST context.
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)

	// Call LLM via stream.
	reader, err := dc.client.GenerateStream(ctx, prompt, currentConfig)
	if err != nil {
		// Error already wrapped by client.
		return "", err
	}

	// Buffer the streaming response.
	var buffer bytes.Buffer
	streamCtx, cancelStream := context.WithTimeout(ctx, 50*time.Second) // Timeout for stream processing.
	defer cancelStream()
	if streamErr := streamCompletion(streamCtx, reader, &buffer); streamErr != nil {
		// Handle stream context errors vs processing errors.
		if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
			return "", fmt.Errorf("%w: streaming context error: %w", ErrOllamaUnavailable, streamErr)
		}
		return "", fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
	}
	return strings.TrimSpace(buffer.String()), nil
}

// GetCompletionStreamFromFile provides context-aware completion using analysis, streaming the result.
func (dc *DeepCompleter) GetCompletionStreamFromFile(
	ctx context.Context,
	filename string, // Use URI as filename for consistency with LSP.
	row, col int, // 1-based line/column.
	w io.Writer, // Writer for streaming output.
) error {
	// Use read lock to access config safely.
	dc.configMu.RLock()
	currentConfig := dc.config // Make a copy while holding lock.
	dc.configMu.RUnlock()

	var contextPreamble string = "// Basic file context only." // Default preamble.
	var analysisInfo *AstContextInfo
	var analysisErr error // Store non-fatal analysis errors.

	// Perform AST analysis if enabled.
	if currentConfig.UseAst {
		log.Printf("Analyzing context (or checking cache) for %s:%d:%d", filename, row, col)
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second) // Timeout for analysis.
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, filename, row, col)
		cancelAnalysis()

		// Check for FATAL analysis errors (errors other than ErrAnalysisFailed/ErrCache).
		// ErrAnalysisFailed itself indicates non-fatal issues occurred.
		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) && !errors.Is(analysisErr, ErrCache) {
			// If Analyze returns a different error type, it's likely fatal (e.g., file read error).
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		// Log non-fatal errors if they occurred.
		if analysisErr != nil {
			// logAnalysisErrors is called within Analyze, but log context here too.
			log.Printf("Non-fatal error during analysis/cache check for %s: %v", filename, analysisErr)
		}

		// Use preamble from analysis results if available.
		if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else if analysisErr != nil {
			// Add warning to preamble if analysis had errors but produced no context.
			contextPreamble += fmt.Sprintf("\n// Warning: Context analysis completed with errors and no preamble: %v\n", analysisErr)
		} else {
			// Add warning if analysis succeeded but produced no context.
			contextPreamble += "\n// Warning: Context analysis returned no specific context preamble.\n"
		}
	} else {
		log.Println("AST analysis disabled by config.")
	}

	// Extract code snippet around the cursor. This can fail (e.g., file not found).
	snippetCtx, err := extractSnippetContext(filename, row, col)
	if err != nil {
		// Snippet extraction failure is fatal for file-based completion.
		return fmt.Errorf("failed to extract code snippet context: %w", err)
	}

	// Format the final prompt using potentially updated preamble and snippet.
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	// Log truncated prompt for debugging.
	if len(prompt) > 1000 {
		log.Printf("Generated Prompt (length %d):\n---\n%s\n... (prompt truncated in log)\n---", len(prompt), prompt[:1000])
	} else {
		log.Printf("Generated Prompt (length %d):\n---\n%s\n---", len(prompt), prompt)
	}

	// --- Call Ollama API with Retry Logic ---
	apiCallFunc := func() error {
		// Use a separate context with timeout for each API attempt.
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()

		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig)
		if apiErr != nil {
			// Check if the error is retryable (e.g., timeout, specific HTTP codes).
			var oe *OllamaError
			isRetryable := errors.As(apiErr, &oe) &&
				(oe.Status == http.StatusServiceUnavailable || oe.Status == http.StatusTooManyRequests)
			isRetryable = isRetryable || errors.Is(apiErr, context.DeadlineExceeded)
			// Return raw error if retryable, wrap if fatal.
			if isRetryable {
				return apiErr // Signal retry logic to retry.
			}
			// Wrap non-retryable client errors. ErrOllamaUnavailable is already wrapped.
			return apiErr
		}

		// Stream the response directly to the provided writer.
		// Add header only if writing to stderr (for CLI). Check writer type?
		// if w == os.Stderr { // This check is fragile. Assume caller handles headers.
		// 	PrettyPrint(ColorGreen, "Completion:\n")
		// }
		streamErr := streamCompletion(apiCtx, reader, w)
		// Add newline after stream if writing to stdout/stderr? Assume caller handles.
		// if w == os.Stdout || w == os.Stderr {
		// 	fmt.Fprintln(w)
		// }

		if streamErr != nil {
			// Check for context errors during streaming (retryable).
			if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
				return streamErr // Retryable context error during streaming.
			}
			// Wrap non-retryable stream processing errors.
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		log.Println("Completion stream finished successfully for this attempt.")
		return nil // Success for this attempt.
	}

	// Execute the API call with retry logic.
	err = retry(ctx, apiCallFunc, maxRetries, retryDelay)
	if err != nil {
		// Check final error type after retries.
		var oe *OllamaError
		if errors.As(err, &oe) || errors.Is(err, ErrOllamaUnavailable) ||
			errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			// Wrap final Ollama/network/timeout errors.
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			return err // Return stream processing error directly.
		}
		// General failure after retries.
		return fmt.Errorf("failed to get completion stream after retries: %w", err)
	}

	// Log analysis errors again as a final warning if completion succeeded.
	if analysisErr != nil {
		log.Printf("Warning: Completion succeeded, but context analysis/cache check encountered non-fatal errors: %v", analysisErr)
	}
	return nil // Overall success.
}

// =============================================================================
// Internal Helpers (Split Point)
// =============================================================================

// ... (streamCompletion, processLine, retry, analysis helpers, etc. follow)
// =============================================================================
// Internal Helpers
// =============================================================================

// --- Ollama Stream Processing Helpers ---

// streamCompletion reads the Ollama stream response and writes it to w.
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer) error {
	defer r.Close()
	reader := bufio.NewReader(r)
	for {
		// Check for context cancellation before reading.
		select {
		case <-ctx.Done():
			log.Println("Context cancelled during streaming")
			return ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')
		if err != nil {
			// Handle EOF cleanly. Process any remaining data before returning.
			if err == io.EOF {
				if len(line) > 0 {
					if procErr := processLine(line, w); procErr != nil {
						return procErr // Return error from final processing.
					}
				}
				return nil // Successful end of stream.
			}
			// Check if error is due to context cancellation after read attempt.
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				// Return other read errors.
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			}
		}
		// Process each line received.
		if procErr := processLine(line, w); procErr != nil {
			return procErr // Return error from line processing.
		}
	}
}

// processLine decodes a single line from the Ollama stream and writes the content.
func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil // Ignore empty lines.
	}
	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		// Tolerate potential non-JSON lines (e.g., debugging info from Ollama?).
		log.Printf("Debug: Ignoring non-JSON line from Ollama stream: %s", string(line))
		return nil
	}
	// Check for errors reported within the stream response itself.
	if resp.Error != "" {
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	}
	// Write the actual completion text chunk.
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		return fmt.Errorf("error writing to output: %w", err)
	}
	return nil
}

// --- Retry Helper ---

// retry executes an operation function with backoff and retry logic.
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration) error {
	var lastErr error
	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		// Check context before each attempt.
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled before attempt %d: %v", i+1, ctx.Err())
			return ctx.Err()
		default:
		}

		lastErr = operation()
		if lastErr == nil {
			return nil // Success.
		}

		// Do not retry if context was cancelled during operation.
		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			log.Printf("Attempt %d failed due to context error: %v. Not retrying.", i+1, lastErr)
			return lastErr
		}

		// Check if the error is considered retryable.
		var ollamaErr *OllamaError
		isRetryable := errors.As(lastErr, &ollamaErr) &&
			(ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		isRetryable = isRetryable || errors.Is(lastErr, ErrOllamaUnavailable) // Also retry general unavailability.

		if !isRetryable {
			log.Printf("Attempt %d failed with non-retryable error: %v", i+1, lastErr)
			return lastErr // Return non-retryable error immediately.
		}

		// Wait before retrying, respecting context cancellation.
		waitDuration := currentDelay
		log.Printf("Attempt %d failed with retryable error: %v. Retrying in %v...", i+1, lastErr, waitDuration)
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled during retry wait: %v", ctx.Err())
			return ctx.Err()
		case <-time.After(waitDuration):
			// Optionally implement exponential backoff: currentDelay *= 2
		}
	}
	log.Printf("Operation failed after %d retries.", maxRetries)
	// Return the last error encountered after all retries failed.
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr)
}

// --- Analysis Helpers ---

// loadPackageInfo loads package information using go/packages.
func loadPackageInfo(ctx context.Context, absFilename string, fset *token.FileSet) (*packages.Package, *ast.File, *token.File, []error) {
	dir := filepath.Dir(absFilename)
	var loadErrors []error
	// Configure packages.Load: request necessary modes for analysis.
	cfg := &packages.Config{
		Context: ctx,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedSyntax |
			packages.NeedTypesInfo | packages.NeedTypesSizes | packages.NeedDeps,
		Dir:  dir,
		Fset: fset,
		// Allow parsing files with errors to proceed with partial ASTs.
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			const mode = parser.ParseComments | parser.AllErrors
			file, err := parser.ParseFile(fset, filename, src, mode)
			// Log parser errors but don't return them here; let packages.Load collect them.
			if err != nil {
				log.Printf("Parser error in %s (proceeding with partial AST): %v", filename, err)
			}
			return file, nil
		},
		Tests: true, // Include test packages.
	}
	pkgs, loadErr := packages.Load(cfg, ".") // Load packages in the target directory.
	if loadErr != nil {
		log.Printf("Error loading packages for %s: %v", dir, loadErr)
		loadErrors = append(loadErrors, fmt.Errorf("package loading failed: %w", loadErr))
	}

	// Collect all errors reported by packages.Load.
	if len(pkgs) > 0 {
		packages.Visit(pkgs, nil, func(pkg *packages.Package) {
			for _, err := range pkg.Errors {
				errMsg := err.Error()
				// Avoid duplicate error messages.
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

	// Find the specific package and AST file corresponding to the target filename.
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
				// Normalize paths for reliable comparison.
				normSyntaxFileName, _ := filepath.Abs(tokenFile.Name())
				if normSyntaxFileName == absFilename {
					// Found the target file's AST. Check for required info.
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

	// If the target file wasn't found in any loaded package.
	if len(loadErrors) == 0 { // Only add this error if no other load errors occurred.
		loadErrors = append(loadErrors, fmt.Errorf("target file %s not found within loaded packages for directory %s", absFilename, dir))
	}
	return nil, nil, nil, loadErrors
}

// findEnclosingPath finds the AST node path from the root to the node enclosing the cursor.
func findEnclosingPath(targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo) []ast.Node {
	if targetFileAST == nil {
		addAnalysisError(info, errors.New("cannot find enclosing path: targetFileAST is nil"))
		return nil
	}
	if !cursorPos.IsValid() {
		addAnalysisError(info, errors.New("cannot find enclosing path: invalid cursor position"))
		return nil
	}
	// Use astutil to find the path.
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		log.Printf("DEBUG: No AST path found enclosing cursor position %v", cursorPos)
	}
	return path
}

// gatherScopeContext walks the enclosing path to find relevant scope information.
func gatherScopeContext(path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo) {
	if fset == nil && path != nil {
		log.Println("Warning: Cannot format receiver type - fset is nil in gatherScopeContext.")
	}

	if path != nil {
		// Traverse path from outermost to innermost to find enclosing elements.
		for i := len(path) - 1; i >= 0; i-- {
			node := path[i]
			switch n := node.(type) {
			case *ast.FuncDecl:
				// Capture the first FuncDecl encountered as the enclosing one.
				if info.EnclosingFuncNode == nil {
					info.EnclosingFuncNode = n
					// Try to format receiver type if present.
					if fset != nil && n.Recv != nil && len(n.Recv.List) > 0 && n.Recv.List[0].Type != nil {
						var buf bytes.Buffer
						// Use go/format to get canonical type string.
						if err := format.Node(&buf, fset, n.Recv.List[0].Type); err == nil {
							info.ReceiverType = buf.String()
						} else {
							log.Printf("Warning: could not format receiver type: %v", err)
							info.ReceiverType = "[error formatting receiver]"
							addAnalysisError(info, fmt.Errorf("receiver format error: %w", err))
						}
					}
				}
				// Try to find the corresponding types.Func object using type info.
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
					if obj, ok := targetPkg.TypesInfo.Defs[n.Name]; ok && obj != nil {
						if fn, ok := obj.(*types.Func); ok {
							// Capture the first types.Func found.
							if info.EnclosingFunc == nil {
								info.EnclosingFunc = fn
								// Add function parameters/results to scope map.
								if sig, ok := fn.Type().(*types.Signature); ok {
									addSignatureToScope(sig, info.VariablesInScope)
								}
							}
						}
					} else {
						// Log if definition not found (only for the first enclosing func node found).
						if info.EnclosingFunc == nil && n.Name != nil {
							addAnalysisError(info, fmt.Errorf("definition for func '%s' not found in TypesInfo", n.Name.Name))
						}
					}
				} else if info.EnclosingFuncNode != nil && info.EnclosingFunc == nil { // Log if type info is generally unavailable.
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
				// Capture the innermost block statement containing the cursor.
				if info.EnclosingBlock == nil {
					info.EnclosingBlock = n
				}
				// Add variables declared in this block to the scope.
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					}
				}
			}
		}
	}

	// Always add package-level scope variables.
	addPackageScope(targetPkg, info)
}

// addPackageScope adds package-level identifiers to the scope map.
func addPackageScope(targetPkg *packages.Package, info *AstContextInfo) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope) // No cursor pos needed for package scope.
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

// addScopeVariables adds identifiers from a types.Scope if declared before cursorPos.
func addScopeVariables(typeScope *types.Scope, cursorPos token.Pos, scopeMap map[string]types.Object) {
	if typeScope == nil {
		return
	}
	for _, name := range typeScope.Names() {
		obj := typeScope.Lookup(name)
		if obj == nil {
			continue
		}
		// Include if cursor position is invalid (package scope) or object declared before cursor.
		include := !cursorPos.IsValid() || !obj.Pos().IsValid() || obj.Pos() < cursorPos

		if include {
			// Add if not already present (inner scopes override outer).
			if _, exists := scopeMap[name]; !exists {
				// Filter to relevant object kinds.
				switch obj.(type) {
				case *types.Var, *types.Const, *types.TypeName, *types.Func, *types.Label, *types.PkgName, *types.Builtin, *types.Nil:
					scopeMap[name] = obj
				}
			}
		}
	}
}

// addSignatureToScope adds named parameters and results to the scope map.
func addSignatureToScope(sig *types.Signature, scopeMap map[string]types.Object) {
	if sig == nil {
		return
	}
	addTupleToScope(sig.Params(), scopeMap)
	addTupleToScope(sig.Results(), scopeMap)
}

// addTupleToScope adds named variables from a types.Tuple to the scope map.
func addTupleToScope(tuple *types.Tuple, scopeMap map[string]types.Object) {
	if tuple == nil {
		return
	}
	for j := 0; j < tuple.Len(); j++ {
		v := tuple.At(j)
		if v != nil && v.Name() != "" { // Only add named variables.
			if _, exists := scopeMap[v.Name()]; !exists {
				scopeMap[v.Name()] = v
			}
		}
	}
}

// findRelevantComments uses ast.CommentMap to find comments near the cursor.
func findRelevantComments(targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo) {
	if targetFileAST == nil || fset == nil {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"))
		return
	}
	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset)
}

// findCommentsWithMap implements the logic to find preceding or enclosing doc comments.
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		return comments
	}

	cursorLine := fset.Position(cursorPos).Line
	foundPreceding := false
	var precedingComments []string

	// Strategy 1: Find comments ending on the line immediately before the cursor.
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
		// Strategy 2: Fallback to Doc comments on the enclosing path.
		if path != nil {
			for i := 0; i < len(path); i++ { // Check innermost nodes first.
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
					goto cleanup // Found doc comment, stop searching path.
				}
			}
		}
	}

cleanup: // Deduplicate comments.
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

// buildPreamble constructs the context string sent to the LLM from analysis info.
func buildPreamble(info *AstContextInfo, qualifier types.Qualifier) string {
	var preamble strings.Builder
	// Internal limit for building, final truncation happens later.
	const internalPreambleLimit = 8192
	currentLen := 0

	// Helper to add string if within limit. Returns false if limit reached.
	addToPreamble := func(s string) bool {
		if currentLen+len(s) < internalPreambleLimit {
			preamble.WriteString(s)
			currentLen += len(s)
			return true
		}
		return false
	}
	// Helper to add truncation marker if within limit.
	addTruncMarker := func(section string) {
		msg := fmt.Sprintf("//   ... (%s truncated)\n", section)
		if currentLen+len(msg) < internalPreambleLimit {
			preamble.WriteString(msg)
			currentLen += len(msg)
		}
	}

	// Build preamble section by section, checking limits.
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
	formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker) // Format scope last.

	return preamble.String()
}

// formatImportsSection formats the import list, respecting limits.
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string)) bool {
	if len(info.Imports) == 0 {
		return true
	}
	if !add("// Imports:\n") {
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
		} // Stop adding but return success.
		path := imp.Path.Value
		name := ""
		if imp.Name != nil {
			name = imp.Name.Name + " "
		}
		line := fmt.Sprintf("//   import %s%s\n", name, path)
		if !add(line) {
			return false
		} // Limit reached.
		count++
	}
	return true
}

// formatEnclosingFuncSection formats the enclosing function/method info.
func formatEnclosingFuncSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool) bool {
	funcOrMethod := "Function"
	receiverStr := ""
	if info.ReceiverType != "" {
		funcOrMethod = "Method"
		receiverStr = fmt.Sprintf("(%s) ", info.ReceiverType)
	}

	var funcHeader string
	if info.EnclosingFunc != nil { // Prefer type info.
		name := info.EnclosingFunc.Name()
		sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		if info.ReceiverType != "" && strings.HasPrefix(sigStr, "func(") {
			sigStr = "func" + strings.TrimPrefix(sigStr, "func")
		}
		funcHeader = fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr)
	} else if info.EnclosingFuncNode != nil { // Fallback to AST.
		name := "[anonymous]"
		if info.EnclosingFuncNode.Name != nil {
			name = info.EnclosingFuncNode.Name.Name
		}
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s(...)\n", funcOrMethod, receiverStr, name)
	} else {
		return true // No enclosing function found.
	}
	return add(funcHeader)
}

// formatCommentsSection formats relevant comments, respecting limits.
func formatCommentsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string)) bool {
	if len(info.CommentsNearCursor) == 0 {
		return true
	}
	if !add("// Relevant Comments:\n") {
		return false
	}
	count := 0
	maxComments := 5
	for _, c := range info.CommentsNearCursor {
		if count >= maxComments {
			addTrunc("comments")
			return true
		}
		// Clean comment markers.
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

// formatCursorContextSection formats specific context like calls, selectors, etc.
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool) bool {
	// --- Function Call Context ---
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

		if sig := info.CallExprFuncType; sig != nil { // Add signature details if available.
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
					isVariadic := sig.Variadic() && i == params.Len()-1
					if i == info.CallArgIndex || (isVariadic && info.CallArgIndex >= i) {
						highlight = " // <-- cursor here"
						if isVariadic {
							highlight += " (variadic)"
						}
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
		return true // Handled call context.
	}

	// --- Selector Expression Context ---
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

		if info.SelectorExprType != nil { // List members if type info available.
			members := listTypeMembers(info.SelectorExprType, info.SelectorExpr.X, qualifier)
			var fields, methods []MemberInfo
			if members != nil {
				for _, m := range members {
					if m.Kind == FieldMember {
						fields = append(fields, m)
					} else if m.Kind == MethodMember {
						methods = append(methods, m)
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
				msg := "//   (No exported fields or methods found)\n"
				if members == nil {
					msg = "//   (Could not determine members)\n"
				}
				if !add(msg) {
					return false
				}
			}
		} else {
			if !add("//   (Cannot list members: type analysis failed for base expression)\n") {
				return false
			}
		}
		return true // Handled selector context.
	}

	// --- Composite Literal Context ---
	if info.CompositeLit != nil {
		typeName := "(unknown - type analysis failed for literal)"
		if info.CompositeLitType != nil {
			typeName = types.TypeString(info.CompositeLitType, qualifier)
		}
		if !add(fmt.Sprintf("// Inside composite literal of type: %s\n", typeName)) {
			return false
		}

		if info.CompositeLitType != nil { // Try to list missing struct fields.
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
		return true // Handled composite literal context.
	}

	// --- Identifier Context ---
	if info.IdentifierAtCursor != nil {
		identName := info.IdentifierAtCursor.Name
		identTypeStr := "(Type unknown)"
		if info.IdentifierType != nil {
			identTypeStr = fmt.Sprintf("(Type: %s)", types.TypeString(info.IdentifierType, qualifier))
		}
		if !add(fmt.Sprintf("// Identifier at cursor: %s %s\n", identName, identTypeStr)) {
			return false
		}
		return true // Handled identifier context.
	}

	return true // No specific cursor context found.
}

// formatScopeSection formats variables/constants/types in scope, respecting limits.
func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string)) bool {
	if len(info.VariablesInScope) == 0 {
		return true
	}
	if !add("// Variables/Constants/Types in Scope:\n") {
		return false
	}
	var items []string
	for name := range info.VariablesInScope {
		obj := info.VariablesInScope[name]
		items = append(items, fmt.Sprintf("//   %s\n", types.ObjectString(obj, qualifier)))
	}
	sort.Strings(items) // Sort for deterministic output.
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

// calculateCursorPos converts 1-based line/col to 0-based token.Pos offset.
func calculateCursorPos(file *token.File, line, col int) (token.Pos, error) {
	if line <= 0 {
		return token.NoPos, fmt.Errorf("%w: line number %d must be >= 1", ErrInvalidPositionInput, line)
	}
	if col <= 0 {
		return token.NoPos, fmt.Errorf("%w: column number %d must be >= 1", ErrInvalidPositionInput, col)
	}
	if file == nil {
		return token.NoPos, errors.New("invalid token.File (nil)")
	}

	fileLineCount := file.LineCount()
	if line > fileLineCount {
		if line == fileLineCount+1 && col == 1 {
			return file.Pos(file.Size()), nil
		} // Allow cursor at start of line after last line.
		return token.NoPos, fmt.Errorf("%w: line number %d exceeds file line count %d", ErrPositionOutOfRange, line, fileLineCount)
	}

	lineStartPos := file.LineStart(line)
	if !lineStartPos.IsValid() {
		return token.NoPos, fmt.Errorf("%w: cannot get start offset for line %d in file '%s'", ErrPositionConversion, line, file.Name())
	}

	lineStartOffset := file.Offset(lineStartPos)
	cursorOffset := lineStartOffset + col - 1 // Calculate target offset.

	// Determine end offset of the line.
	lineEndOffset := file.Size()
	if line < fileLineCount {
		nextLineStartPos := file.LineStart(line + 1)
		if nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	// Clamp offset to line boundaries.
	finalOffset := cursorOffset
	if cursorOffset < lineStartOffset {
		finalOffset = lineStartOffset
		log.Printf("Warning: column %d resulted in offset %d before line start %d. Clamping.", col, cursorOffset, lineStartOffset)
	}
	if cursorOffset > lineEndOffset {
		finalOffset = lineEndOffset
		log.Printf("Warning: column %d resulted in offset %d beyond line end %d. Clamping.", col, cursorOffset, lineEndOffset)
	}

	pos := file.Pos(finalOffset)
	if !pos.IsValid() { // Fallback if clamped offset is still invalid.
		log.Printf("Error: Clamped offset %d resulted in invalid token.Pos. Using line start %d.", finalOffset, lineStartPos)
		return lineStartPos, fmt.Errorf("%w: failed to calculate valid token.Pos for offset %d", ErrPositionConversion, finalOffset)
	}
	return pos, nil
}

// findContextNodes identifies specific AST nodes (call, selector, literal, ident) at/near the cursor.
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

	// Check innermost node first.
	innermostNode := path[0]
	// 1. Composite Literal?
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
		} else if hasTypeInfo {
			addAnalysisError(info, errors.New("cannot analyze composite literal: Types map is nil"))
		}
		return
	}
	// 2. Function Call? Check path[0] and path[1].
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
		} else if hasTypeInfo {
			addAnalysisError(info, errors.New("cannot analyze call expr: Types map is nil"))
		}
		return
	}
	// 3. Selector Expression? Check path[0] and path[1].
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
			} else if hasTypeInfo {
				addAnalysisError(info, errors.New("cannot analyze selector expr: Types map is nil"))
			}
			return
		}
	}
	// 4. Identifier? Check path[0] and path[1].
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

// calculateArgIndex determines the 0-based index of the argument the cursor is in.
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
		} // Cursor within arg or its preceding comma space.
		if cursorPos > argEnd { // Cursor is after this argument.
			if i == len(args)-1 {
				return i + 1
			} // After last arg.
			if args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i + 1
			} // Between this arg and next.
		}
	}
	if len(args) > 0 && args[0] != nil && cursorPos < args[0].Pos() {
		return 0
	} // Before first arg.
	return 0 // Default fallback.
}

// determineExpectedArgType finds the expected type for a given argument index in a signature.
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
		if argIndex >= numParams-1 { // At or after variadic parameter.
			lastParam := params.At(numParams - 1)
			if lastParam == nil {
				return nil
			}
			if slice, ok := lastParam.Type().(*types.Slice); ok {
				return slice.Elem()
			} // Return element type.
			return nil
		}
		// Before variadic parameter.
		param := params.At(argIndex)
		if param == nil {
			return nil
		}
		return param.Type()
	} else { // Non-variadic.
		if argIndex < numParams {
			param := params.At(argIndex)
			if param == nil {
				return nil
			}
			return param.Type()
		}
	}
	return nil // Index out of bounds.
}

// listTypeMembers attempts to list exported fields and methods for a given type.
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

	// Handle members based on underlying type kind.
	switch u := underlying.(type) {
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
		// Basic, Map, Slice, Chan: No user-defined members to list here.
	}

	// Add methods from the method set of the original type (value or pointer receiver).
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

	// If original type was not a pointer, also check pointer receiver methods.
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

	// Deduplicate members.
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

// listStructFields lists exported fields of a struct type.
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

// logAnalysisErrors logs joined non-fatal analysis errors.
func logAnalysisErrors(errs []error) {
	if len(errs) > 0 {
		combinedErr := errors.Join(errs...)
		log.Printf("Context analysis completed with %d non-fatal error(s): %v", len(errs), combinedErr)
	}
}

// addAnalysisError adds a non-fatal error, avoiding duplicates based on message.
func addAnalysisError(info *AstContextInfo, err error) {
	if err != nil && info != nil {
		errMsg := err.Error()
		for _, existing := range info.AnalysisErrors {
			if existing.Error() == errMsg {
				return
			}
		} // Skip duplicates.
		log.Printf("Analysis Warning: %v", err) // Log the warning.
		info.AnalysisErrors = append(info.AnalysisErrors, err)
	}
}

// ============================================================================
// LSP Position Conversion Helpers
// ============================================================================

// LspPositionToBytePosition converts 0-based LSP line/character (UTF-16) to
// 1-based Go line/column (bytes) and 0-based byte offset.
func LspPositionToBytePosition(content []byte, lspPos LSPPosition) (line, col, byteOffset int, err error) {
	if content == nil {
		return 0, 0, -1, fmt.Errorf("%w: file content is nil", ErrPositionConversion)
	}
	targetLine := int(lspPos.Line)
	targetUTF16Char := int(lspPos.Character)
	if targetLine < 0 {
		return 0, 0, -1, fmt.Errorf("%w: line number %d must be >= 0", ErrInvalidPositionInput, targetLine)
	}
	if targetUTF16Char < 0 {
		return 0, 0, -1, fmt.Errorf("%w: character offset %d must be >= 0", ErrInvalidPositionInput, targetUTF16Char)
	}

	currentLine := 0
	currentByteOffset := 0
	scanner := bufio.NewScanner(bytes.NewReader(content))
	for scanner.Scan() {
		lineTextBytes := scanner.Bytes()
		lineLengthBytes := len(lineTextBytes)
		newlineLengthBytes := 1
		if currentLine == targetLine {
			byteOffsetInLine, convErr := Utf16OffsetToBytes(lineTextBytes, targetUTF16Char)
			if convErr != nil {
				if errors.Is(convErr, ErrPositionOutOfRange) { // Clamp to line end on out-of-range error.
					log.Printf("Warning: utf16OffsetToBytes reported offset out of range (line %d, char %d): %v. Clamping to line end.", targetLine, targetUTF16Char, convErr)
					byteOffsetInLine = lineLengthBytes
				} else {
					return 0, 0, -1, fmt.Errorf("failed converting UTF16 to byte offset on line %d: %w", currentLine, convErr)
				}
			}
			line = currentLine + 1
			col = byteOffsetInLine + 1
			byteOffset = currentByteOffset + byteOffsetInLine
			return line, col, byteOffset, nil // Success.
		}
		currentByteOffset += lineLengthBytes + newlineLengthBytes
		currentLine++
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, -1, fmt.Errorf("%w: error scanning file content: %w", ErrPositionConversion, err)
	}

	// Handle cursor on the line after the last line of content.
	if currentLine == targetLine {
		if targetUTF16Char == 0 {
			line = currentLine + 1
			col = 1
			byteOffset = currentByteOffset
			return line, col, byteOffset, nil
		}
		return 0, 0, -1, fmt.Errorf("%w: invalid character offset %d on line %d (after last line with content)", ErrPositionOutOfRange, targetUTF16Char, targetLine)
	}
	// Target line not found.
	return 0, 0, -1, fmt.Errorf("%w: LSP line %d not found in file (total lines scanned %d)", ErrPositionOutOfRange, targetLine, currentLine)
}

// Utf16OffsetToBytes converts a 0-based UTF-16 offset within a line to a 0-based byte offset.
func Utf16OffsetToBytes(line []byte, utf16Offset int) (int, error) {
	if utf16Offset < 0 {
		return 0, fmt.Errorf("%w: invalid utf16Offset: %d (must be >= 0)", ErrInvalidPositionInput, utf16Offset)
	}
	if utf16Offset == 0 {
		return 0, nil
	}

	byteOffset := 0
	currentUTF16Offset := 0
	for byteOffset < len(line) {
		if currentUTF16Offset >= utf16Offset {
			break
		} // Reached target.
		r, size := utf8.DecodeRune(line[byteOffset:])
		if r == utf8.RuneError && size <= 1 {
			return byteOffset, fmt.Errorf("%w at byte offset %d", ErrInvalidUTF8, byteOffset)
		}
		utf16Units := 1
		if r > 0xFFFF {
			utf16Units = 2
		} // Surrogate pairs require 2 units.
		// If adding this rune exceeds target, current byteOffset is the answer.
		if currentUTF16Offset+utf16Units > utf16Offset {
			break
		}
		currentUTF16Offset += utf16Units
		byteOffset += size
		if currentUTF16Offset == utf16Offset {
			break
		} // Exact match.
	}
	// Check if target offset was beyond the actual line length in UTF-16.
	if currentUTF16Offset < utf16Offset {
		return len(line), fmt.Errorf("%w: utf16Offset %d is beyond the line length in UTF-16 units (%d)", ErrPositionOutOfRange, utf16Offset, currentUTF16Offset)
	}
	return byteOffset, nil
}

// ============================================================================
// Cache Helper Functions
// ============================================================================

// calculateGoModHash calculates the SHA256 hash of the go.mod file.
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

// calculateInputHashes calculates hashes for go.mod, go.sum, and Go files.
func calculateInputHashes(dir string, pkg *packages.Package) (map[string]string, error) {
	hashes := make(map[string]string)
	filesToHash := make(map[string]struct{})

	// Always check for go.mod/go.sum.
	for _, fname := range []string{"go.mod", "go.sum"} {
		fpath := filepath.Join(dir, fname)
		if _, err := os.Stat(fpath); err == nil {
			if absPath, absErr := filepath.Abs(fpath); absErr == nil {
				filesToHash[absPath] = struct{}{}
			} else {
				log.Printf("Warning: Could not get absolute path for %s: %v", fpath, absErr)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("failed to stat %s: %w", fpath, err)
		}
	}

	// Use compiled files from package info if available.
	filesFromPkg := false
	if pkg != nil && len(pkg.CompiledGoFiles) > 0 {
		filesFromPkg = true
		log.Printf("DEBUG: Hashing based on %d CompiledGoFiles from package %s", len(pkg.CompiledGoFiles), pkg.PkgPath)
		for _, fpath := range pkg.CompiledGoFiles {
			if absPath, err := filepath.Abs(fpath); err == nil {
				filesToHash[absPath] = struct{}{}
			} else {
				log.Printf("Warning: Could not get absolute path for compiled file %s: %v", fpath, err)
			}
		}
	}

	// Fallback: Scan directory for .go files if package info unavailable/empty.
	if !filesFromPkg {
		log.Printf("DEBUG: Calculating input hashes by scanning directory %s (pkg info unavailable or empty)", dir)
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, fmt.Errorf("failed to scan directory %s for hashing: %w", dir, err)
		}
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") {
				absPath := filepath.Join(dir, entry.Name())
				if absPath, absErr := filepath.Abs(absPath); absErr == nil {
					filesToHash[absPath] = struct{}{}
				} else {
					log.Printf("Warning: Could not get absolute path for %s: %v", entry.Name(), absErr)
				}
			}
		}
	}

	// Calculate hash for each unique file path.
	for absPath := range filesToHash {
		relPath, err := filepath.Rel(dir, absPath)
		if err != nil {
			log.Printf("Warning: Could not get relative path for %s in %s: %v. Using base name.", absPath, dir, err)
			relPath = filepath.Base(absPath)
		}
		relPath = filepath.ToSlash(relPath) // Use forward slashes for consistent keys.

		hash, err := hashFileContent(absPath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				log.Printf("Warning: File %s disappeared during hashing. Skipping.", absPath)
				continue
			}
			return nil, fmt.Errorf("%w: failed to hash input file %s: %w", ErrCacheHash, absPath, err)
		}
		hashes[relPath] = hash
	}
	log.Printf("DEBUG: Calculated hashes for %d input files in %s", len(hashes), dir)
	return hashes, nil
}

// hashFileContent calculates the SHA256 hash of a single file.
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

// compareFileHashes compares current and cached file hashes, logging differences.
func compareFileHashes(current, cached map[string]string) bool {
	if len(current) != len(cached) {
		log.Printf("DEBUG: Cache invalid: File count mismatch (Current: %d, Cached: %d)", len(current), len(cached))
		for relPath := range current {
			if _, ok := cached[relPath]; !ok {
				log.Printf("DEBUG:   - File '%s' exists now but was not in cache.", relPath)
			}
		}
		for relPath := range cached {
			if _, ok := current[relPath]; !ok {
				log.Printf("DEBUG:   - File '%s' was cached but does not exist now.", relPath)
			}
		}
		return false
	}
	for relPath, currentHash := range current {
		cachedHash, ok := cached[relPath]
		if !ok {
			log.Printf("DEBUG: Cache invalid: File '%s' missing in cache despite matching counts.", relPath)
			return false
		} // Should not happen if counts match.
		if currentHash != cachedHash {
			log.Printf("DEBUG: Cache invalid: Hash mismatch for file '%s'.", relPath)
			return false
		}
	}
	return true // Counts match and all hashes match.
}

// deleteCacheEntry removes an entry from the bbolt cache.
func (a *GoPackagesAnalyzer) deleteCacheEntry(cacheKey []byte) {
	if a.db == nil {
		return
	} // Cache disabled.
	err := a.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(cacheBucketName)
		if b == nil {
			return nil
		} // Bucket gone? Nothing to do.
		if b.Get(cacheKey) == nil {
			return nil
		} // Key doesn't exist.
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

// Spinner provides simple terminal spinner feedback.
type Spinner struct {
	chars    []string
	message  string
	index    int
	mu       sync.Mutex
	stopChan chan struct{}
	doneChan chan struct{}
	running  bool
}

func NewSpinner() *Spinner {
	return &Spinner{chars: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}, index: 0}
}

// Start begins the spinner animation in a separate goroutine.
func (s *Spinner) Start(initialMessage string) {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.stopChan = make(chan struct{})
	s.doneChan = make(chan struct{})
	s.message = initialMessage
	s.running = true
	s.mu.Unlock()
	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		defer func() { // Cleanup function.
			s.mu.Lock()
			isRunning := s.running
			s.running = false
			s.mu.Unlock()
			if isRunning {
				fmt.Fprintf(os.Stderr, "\r\033[K")
			} // Clear line only if running.
			select {
			case s.doneChan <- struct{}{}:
			default:
			} // Signal done.
			close(s.doneChan)
		}()
		for {
			select {
			case <-s.stopChan:
				return // Exit on stop signal.
			case <-ticker.C:
				s.mu.Lock()
				if !s.running {
					s.mu.Unlock()
					return
				} // Check running status again.
				char := s.chars[s.index]
				msg := s.message
				s.index = (s.index + 1) % len(s.chars)
				fmt.Fprintf(os.Stderr, "\r\033[K%s%s%s %s", ColorCyan, char, ColorReset, msg) // Update spinner line.
				s.mu.Unlock()
			}
		}
	}()
}

// UpdateMessage changes the text displayed next to the spinner.
func (s *Spinner) UpdateMessage(newMessage string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		s.message = newMessage
	}
}

// Stop halts the spinner animation and cleans up.
func (s *Spinner) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	select {
	case <-s.stopChan:
	default:
		close(s.stopChan)
	} // Close stopChan safely.
	doneChan := s.doneChan
	s.mu.Unlock()
	if doneChan != nil { // Wait for goroutine cleanup with timeout.
		select {
		case <-doneChan:
		case <-time.After(500 * time.Millisecond):
			log.Println("Warning: Timeout waiting for spinner goroutine cleanup")
		}
	}
	fmt.Fprintf(os.Stderr, "\r\033[K") // Final clear line.
}

// ============================================================================
// Snippet Extraction Helper
// ============================================================================

// extractSnippetContext extracts code prefix, suffix, and full line around the cursor.
func extractSnippetContext(filename string, row, col int) (SnippetContext, error) {
	var ctx SnippetContext
	contentBytes, err := os.ReadFile(filename)
	if err != nil {
		return ctx, fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	content := string(contentBytes)

	fset := token.NewFileSet()
	file := fset.AddFile(filename, 1, len(contentBytes)) // Base 1, correct size.
	if file == nil {
		return ctx, fmt.Errorf("failed to add file '%s' to fileset", filename)
	}

	// Calculate 0-based byte offset.
	cursorPos, posErr := calculateCursorPos(file, row, col)
	if posErr != nil {
		return ctx, fmt.Errorf("cannot determine valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		return ctx, fmt.Errorf("%w: invalid cursor position calculated (Pos: %d)", ErrPositionConversion, cursorPos)
	}

	offset := file.Offset(cursorPos)
	// Clamp offset to content bounds.
	if offset < 0 {
		offset = 0
	}
	if offset > len(content) {
		offset = len(content)
	}

	ctx.Prefix = content[:offset]
	ctx.Suffix = content[offset:]

	// Extract the full line content.
	lineStartPos := file.LineStart(row)
	if !lineStartPos.IsValid() {
		log.Printf("Warning: Could not get start position for line %d in %s", row, filename)
		return ctx, nil
	}
	startOffset := file.Offset(lineStartPos)

	lineEndOffset := file.Size()
	fileLineCount := file.LineCount()
	if row < fileLineCount {
		if nextLineStartPos := file.LineStart(row + 1); nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	if startOffset >= 0 && lineEndOffset >= startOffset && lineEndOffset <= len(content) {
		lineContent := content[startOffset:lineEndOffset]
		// Trim trailing newline characters.
		lineContent = strings.TrimRight(lineContent, "\n")
		lineContent = strings.TrimRight(lineContent, "\r")
		ctx.FullLine = lineContent
	} else {
		log.Printf("Warning: Could not extract full line for row %d (start %d, end %d, content len %d) in %s", row, startOffset, lineEndOffset, len(content), filename)
	}
	return ctx, nil
}
