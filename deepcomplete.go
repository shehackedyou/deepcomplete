// deepcomplete.go
// Package deepcomplete provides core logic for local code completion using LLMs.
package deepcomplete

import (
	"bytes"
	"context"
	"encoding/gob" // For cache serialization
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
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

	"go.etcd.io/bbolt"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

// =============================================================================
// Constants & Core Types (Remain in main package file)
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

	defaultMaxTokens      = 256            // Default maximum tokens for LLM response.
	DefaultStop           = "\n"           // Default stop sequence for LLM. Exported for CLI use.
	defaultTemperature    = 0.1            // Default sampling temperature for LLM.
	defaultConfigFileName = "config.json"  // Default config file name.
	configDirName         = "deepcomplete" // Subdirectory name for config/data.
	cacheSchemaVersion    = 2              // Used to invalidate cache if internal formats change.

	// Phase 1 Fix: Define retry constants
	maxRetries = 3
	retryDelay = 500 * time.Millisecond
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
type AstContextInfo struct {
	FilePath           string
	CursorPos          token.Pos
	PackageName        string
	EnclosingFunc      *types.Func
	EnclosingFuncNode  *ast.FuncDecl
	ReceiverType       string
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
	PromptPreamble     string
	AnalysisErrors     []error
}

// OllamaError defines a custom error for Ollama API issues.
type OllamaError struct {
	Message string
	Status  int
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
	SchemaVersion   int
	GoModHash       string
	InputFileHashes map[string]string // key: relative path (using '/')
	AnalysisGob     []byte
}

// MemberKind defines the type of member (field or method).
type MemberKind string

const (
	FieldMember  MemberKind = "field"
	MethodMember MemberKind = "method"
	OtherMember  MemberKind = "other"
)

// MemberInfo holds structured information about a type member.
type MemberInfo struct {
	Name       string
	Kind       MemberKind
	TypeString string
}

// =============================================================================
// Exported Errors (Remain in main package file)
// =============================================================================

var (
	ErrAnalysisFailed       = errors.New("code analysis failed")
	ErrOllamaUnavailable    = errors.New("ollama API unavailable")
	ErrStreamProcessing     = errors.New("error processing LLM stream")
	ErrConfig               = errors.New("configuration error")
	ErrInvalidConfig        = errors.New("invalid configuration")
	ErrCache                = errors.New("cache operation failed")
	ErrCacheRead            = errors.New("cache read failed")
	ErrCacheWrite           = errors.New("cache write failed")
	ErrCacheDecode          = errors.New("cache decode failed")
	ErrCacheEncode          = errors.New("cache encode failed")
	ErrCacheHash            = errors.New("cache hash calculation failed")
	ErrPositionConversion   = errors.New("position conversion failed")
	ErrInvalidPositionInput = errors.New("invalid input position")
	ErrPositionOutOfRange   = errors.New("position out of range")
	ErrInvalidUTF8          = errors.New("invalid utf-8 sequence")
)

// =============================================================================
// Interfaces for Components (Remain in main package file)
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
// Variables & Default Config (Remain in main package file)
// =============================================================================

var (
	// DefaultConfig provides default settings.
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
)

// =============================================================================
// Configuration Loading (Remains with Config type)
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
		} else if loadedFromFile {
			log.Printf("Attempted load from %s but failed.", primaryPath)
		}
	}

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
		} else if loadedFromFile {
			log.Printf("Attempted load from %s but failed.", secondaryPath)
		}
	}

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
		cfg = DefaultConfig
	}

	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	finalCfg := cfg
	if err := finalCfg.Validate(); err != nil {
		log.Printf("Warning: Config after load/merge failed validation: %v. Returning defaults.", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		if valErr := DefaultConfig.Validate(); valErr != nil {
			return DefaultConfig, fmt.Errorf("default config is invalid: %w", valErr)
		}
		finalCfg = DefaultConfig
	}

	if len(loadErrors) > 0 {
		return finalCfg, fmt.Errorf("%w: %w", ErrConfig, errors.Join(loadErrors...))
	}
	return finalCfg, nil
}

// getConfigPaths determines the primary (XDG) and secondary (~/.config) config paths.
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
		secondary = filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
		if primary == "" && cfgErr != nil {
			primary = secondary
			log.Printf("Using fallback primary config path: %s", primary)
			secondary = ""
		}
		if primary == secondary {
			secondary = ""
		}
	} else {
		log.Printf("Warning: Could not determine user home directory: %v", homeErr)
	}
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
	expCfg := ExportableConfig{OllamaURL: defaultConfig.OllamaURL, Model: defaultConfig.Model, MaxTokens: defaultConfig.MaxTokens, Stop: defaultConfig.Stop, Temperature: defaultConfig.Temperature, UseAst: defaultConfig.UseAst, UseFim: defaultConfig.UseFim, MaxPreambleLen: defaultConfig.MaxPreambleLen, MaxSnippetLen: defaultConfig.MaxSnippetLen}
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
// Default Component Implementations (Remain in main package file)
// =============================================================================

// httpOllamaClient implements LLMClient using HTTP requests to Ollama.
type httpOllamaClient struct {
	httpClient *http.Client
}

func newHttpOllamaClient() *httpOllamaClient {
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

	payload := map[string]interface{}{"model": config.Model, "prompt": prompt, "stream": true, "options": map[string]interface{}{"temperature": config.Temperature, "num_ctx": 4096, "top_p": 0.9, "stop": config.Stop, "num_predict": config.MaxTokens}}
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
		return nil, fmt.Errorf("%w: http request failed: %w", ErrOllamaUnavailable, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(failed to read error response body)"
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

// GoPackagesAnalyzer implements Analyzer using go/packages and bbolt caching.
type GoPackagesAnalyzer struct {
	db *bbolt.DB
	mu sync.Mutex // Protects access to db handle during Close.
}

// NewGoPackagesAnalyzer initializes the analyzer and opens the bbolt cache DB.
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

// Close closes the bbolt database connection if open.
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

// Analyze performs code analysis, utilizing the cache if possible.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, filename string, line, col int) (info *AstContextInfo, analysisErr error) {
	info = &AstContextInfo{FilePath: filename, VariablesInScope: make(map[string]types.Object), AnalysisErrors: make([]error, 0), CallArgIndex: -1}
	defer func() { // Recover from panics during analysis.
		if r := recover(); r != nil {
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

	// --- Cache Check ---
	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	var loadDuration, stepsDuration, preambleDuration time.Duration
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry

	if a.db != nil { // Try reading from cache.
		readStart := time.Now()
		dbErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				return fmt.Errorf("%w: cache bucket %s not found during View", ErrCacheRead, string(cacheBucketName))
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				return nil /* Cache miss */
			}
			var decoded CachedAnalysisEntry
			decoder := gob.NewDecoder(bytes.NewReader(valBytes))
			if err := decoder.Decode(&decoded); err != nil {
				log.Printf("Warning: Failed to gob-decode cached entry header for key %s: %v. Treating as miss.", string(cacheKey), err)
				deleteErr := deleteCacheEntryByKey(a.db, cacheKey)
				if deleteErr != nil {
					log.Printf("Warning: Failed to delete invalid cache entry %s: %v", string(cacheKey), deleteErr)
				}
				return fmt.Errorf("%w: %w", ErrCacheDecode, err)
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				log.Printf("Warning: Cache data for key %s has old schema version %d (want %d). Ignoring.", string(cacheKey), decoded.SchemaVersion, cacheSchemaVersion)
				deleteErr := deleteCacheEntryByKey(a.db, cacheKey)
				if deleteErr != nil {
					log.Printf("Warning: Failed to delete invalid cache entry %s: %v", string(cacheKey), deleteErr)
				}
				return nil /* Treat as miss */
			}
			cachedEntry = &decoded
			return nil
		})
		if dbErr != nil {
			log.Printf("Warning: Error reading or decoding from bbolt cache: %v", dbErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbErr))
			// No need to delete here, already handled in decode error cases above.
		}
		log.Printf("DEBUG: Cache read attempt took %v", time.Since(readStart))
	} else {
		log.Printf("DEBUG: Cache disabled (db handle is nil).")
	}

	// Validate cache hit based on file hashes.
	if cachedEntry != nil {
		validationStart := time.Now()
		log.Printf("DEBUG: Potential cache hit for key: %s. Validating file hashes...", string(cacheKey))
		// Phase 2, Step 4: Pass nil for pkg during validation hash calculation
		currentHashes, hashErr := calculateInputHashes(dir, nil)
		if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes) {
			log.Printf("DEBUG: Cache VALID for key: %s. Attempting to decode analysis data...", string(cacheKey))
			decodeStart := time.Now()
			var analysisData CachedAnalysisData
			decoder := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob))
			if decodeErr := decoder.Decode(&analysisData); decodeErr == nil {
				info.PackageName = analysisData.PackageName
				info.PromptPreamble = analysisData.PromptPreamble
				cacheHit = true
				loadDuration = time.Since(decodeStart)
				log.Printf("DEBUG: Analysis data successfully decoded from cache in %v.", loadDuration)
				log.Printf("DEBUG: Using cached preamble (length %d). Skipping packages.Load and analysis steps.", len(info.PromptPreamble))
			} else {
				log.Printf("Warning: Failed to gob-decode cached analysis data: %v. Treating as miss.", decodeErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheDecode, decodeErr))
				deleteErr := deleteCacheEntryByKey(a.db, cacheKey)
				if deleteErr != nil {
					log.Printf("Warning: Failed to delete invalid cache entry %s: %v", string(cacheKey), deleteErr)
				}
			}
		} else {
			log.Printf("DEBUG: Cache INVALID for key: %s (HashErr: %v). Treating as miss.", string(cacheKey), hashErr)
			deleteErr := deleteCacheEntryByKey(a.db, cacheKey)
			if deleteErr != nil {
				log.Printf("Warning: Failed to delete invalid cache entry %s: %v", string(cacheKey), deleteErr)
			}
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

		loadStart := time.Now()
		fset := token.NewFileSet()
		targetPkg, targetFileAST, targetFile, loadErrors := loadPackageInfo(ctx, absFilename, fset)
		loadDuration = time.Since(loadStart)
		log.Printf("DEBUG: packages.Load completed in %v", loadDuration)
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr)
		}

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

		var qualifier types.Qualifier // Define qualifier here.
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

		// Save to cache if enabled, successful, and preamble generated.
		if a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0 {
			log.Printf("DEBUG: Attempting to save analysis results to bbolt cache. Key: %s", string(cacheKey))
			saveStart := time.Now()
			// Phase 2, Step 4: Pass loaded targetPkg to calculateInputHashes for saving.
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg)
			if hashErr == nil {
				analysisDataToCache := CachedAnalysisData{PackageName: info.PackageName, PromptPreamble: info.PromptPreamble}
				var gobBuf bytes.Buffer
				encoder := gob.NewEncoder(&gobBuf)
				if encodeErr := encoder.Encode(&analysisDataToCache); encodeErr == nil {
					analysisGob := gobBuf.Bytes()
					entryToSave := CachedAnalysisEntry{SchemaVersion: cacheSchemaVersion, GoModHash: goModHash, InputFileHashes: inputHashes, AnalysisGob: analysisGob}
					var entryBuf bytes.Buffer
					entryEncoder := gob.NewEncoder(&entryBuf)
					if entryEncodeErr := entryEncoder.Encode(&entryToSave); entryEncodeErr == nil {
						encodedBytes := entryBuf.Bytes()
						saveErr := a.db.Update(func(tx *bbolt.Tx) error {
							b := tx.Bucket(cacheBucketName)
							if b == nil {
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
			log.Printf("DEBUG: Skipping cache save for key %s (Load Errors: %d, Preamble Empty: %t)", string(cacheKey), len(loadErrors), info.PromptPreamble == "")
		}
	}

	logAnalysisErrors(info.AnalysisErrors)
	analysisErr = errors.Join(info.AnalysisErrors...)
	if cacheHit {
		log.Printf("Context analysis finished (Decode: %v, Cache Hit: %t)", loadDuration, cacheHit)
	} else {
		log.Printf("Context analysis finished (Load: %v, Steps: %v, Preamble: %v, Cache Hit: %t)", loadDuration, stepsDuration, preambleDuration, cacheHit)
	}
	if len(info.PromptPreamble) > 500 {
		log.Printf("Final Context Preamble (length %d):\n---\n%s\n... (preamble truncated in log)\n---", len(info.PromptPreamble), info.PromptPreamble[:500])
	} else {
		log.Printf("Final Context Preamble (length %d):\n---\n%s\n---", len(info.PromptPreamble), info.PromptPreamble)
	}

	// Return non-fatal analysis errors wrapped.
	if analysisErr != nil {
		return info, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisErr)
	}
	return info, nil
}

// InvalidateCache removes the cached entry for a given directory.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	// Phase 2, Step 5: No changes needed here, already correct from Phase 1.
	a.mu.Lock()
	db := a.db
	a.mu.Unlock()
	if db == nil {
		log.Printf("DEBUG: Cache invalidation skipped: DB is nil.")
		return nil
	}
	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	return deleteCacheEntryByKey(db, cacheKey)
}

// loadPackageInfo encapsulates the logic for loading package information.
func loadPackageInfo(ctx context.Context, absFilename string, fset *token.FileSet) (*packages.Package, *ast.File, *token.File, []error) {
	var loadErrors []error
	dir := filepath.Dir(absFilename)

	// Configuration for go/packages
	loadCfg := &packages.Config{
		Context: ctx,
		Dir:     dir,
		Fset:    fset,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedTypesSizes |
			packages.NeedSyntax | packages.NeedTypesInfo,
		Tests: false, // Initially don't load test packages, adjust if needed.
	}

	pkgs, err := packages.Load(loadCfg, fmt.Sprintf("file=%s", absFilename))
	if err != nil {
		loadErrors = append(loadErrors, fmt.Errorf("packages.Load failed: %w", err))
		log.Printf("Error during packages.Load: %v", err)
		return nil, nil, nil, loadErrors
	}

	if len(pkgs) == 0 {
		loadErrors = append(loadErrors, errors.New("packages.Load returned no packages"))
		log.Println("Warning: packages.Load returned no packages.")
		return nil, nil, nil, loadErrors
	}

	var targetPkg *packages.Package
	var targetFileAST *ast.File
	var targetFile *token.File

	// Collect all errors from loaded packages.
	for _, p := range pkgs {
		for _, loadErr := range p.Errors {
			loadErrors = append(loadErrors, &loadErr)
			log.Printf("Package loading error (%s): %v", p.PkgPath, loadErr)
		}
	}

	// Find the package and file containing the target filename.
	for _, p := range pkgs {
		if p == nil {
			continue
		}
		for _, astFile := range p.Syntax {
			if astFile == nil {
				continue
			}
			filePos := fset.Position(astFile.Pos())
			if filePos.IsValid() && filePos.Filename == absFilename {
				targetPkg = p
				targetFileAST = astFile
				targetFile = fset.File(astFile.Pos())
				log.Printf("Found target file %s in package %s", absFilename, p.PkgPath)
				goto foundTarget
			}
		}
	}

foundTarget:
	if targetPkg == nil {
		loadErrors = append(loadErrors, fmt.Errorf("target file %s not found in loaded packages", absFilename))
		log.Printf("Warning: Target file %s not found in any loaded packages.", absFilename)
		// Fallback: Return the first package if available, though AST/Token file will be nil.
		if len(pkgs) > 0 {
			targetPkg = pkgs[0]
			log.Printf("Warning: Falling back to first loaded package %s, but target file AST/Token info will be missing.", targetPkg.PkgPath)
		}
	}

	// Perform basic nil checks on the result.
	if targetPkg != nil {
		if targetPkg.TypesInfo == nil {
			loadErrors = append(loadErrors, fmt.Errorf("type info (TypesInfo) is nil for package %s", targetPkg.PkgPath))
		}
		if targetPkg.Types == nil {
			loadErrors = append(loadErrors, fmt.Errorf("types (Types) is nil for package %s", targetPkg.PkgPath))
		}
	}

	return targetPkg, targetFileAST, targetFile, loadErrors
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
	cursorPos, posErr := calculateCursorPos(targetFile, line, col) // Use helper from utils.go
	if posErr != nil {
		return fmt.Errorf("%w: %w", ErrPositionConversion, posErr)
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
		gatherScopeContext(nil, targetPkg, fset, info)
	}
	return nil
}

// --- Prompt Formatter ---
type templateFormatter struct{}

func newTemplateFormatter() *templateFormatter { return &templateFormatter{} }

// FormatPrompt combines context and snippet into the final LLM prompt, applying truncation.
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2

	if len(contextPreamble) > maxPreambleLen {
		log.Printf("Warning: Truncating context preamble from %d to %d bytes.", len(contextPreamble), maxPreambleLen)
		marker := "... (context truncated)\n"
		startByte := len(contextPreamble) - maxPreambleLen + len(marker)
		if startByte < 0 {
			startByte = 0
		}
		contextPreamble = marker + contextPreamble[startByte:]
	}

	if config.UseFim {
		template = config.FimTemplate
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix
		if len(prefix) > maxFIMPartLen {
			log.Printf("Warning: Truncating FIM prefix from %d to %d bytes.", len(prefix), maxFIMPartLen)
			marker := "...(prefix truncated)"
			startByte := len(prefix) - maxFIMPartLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			prefix = marker + prefix[startByte:]
		}
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
	} else {
		snippet := snippetCtx.Prefix
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
// DeepCompleter Service (Remains in main package file)
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
	cfg, configErr := LoadConfig()
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		return nil, configErr
	}
	if configErr != nil {
		log.Printf("Warning during initial config load: %v", configErr)
	}
	if err := cfg.Validate(); err != nil {
		log.Printf("Warning: Initial config is invalid after load/merge: %v. Using pure defaults.", err)
		cfg = DefaultConfig
		if valErr := cfg.Validate(); valErr != nil {
			return nil, fmt.Errorf("default config validation failed: %w", valErr)
		}
	}
	analyzer := NewGoPackagesAnalyzer()
	dc := &DeepCompleter{client: newHttpOllamaClient(), analyzer: analyzer, formatter: newTemplateFormatter(), config: cfg}
	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with provided config.
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
	return &DeepCompleter{client: newHttpOllamaClient(), analyzer: analyzer, formatter: newTemplateFormatter(), config: config}, nil
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
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate
	}
	if err := newConfig.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	dc.configMu.Lock()
	defer dc.configMu.Unlock()
	dc.config = newConfig
	log.Printf("DeepCompleter configuration updated: %+v", newConfig)
	return nil
}

// GetCompletion provides basic completion for a direct code snippet (no AST analysis).
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	log.Println("DeepCompleter.GetCompletion called for basic prompt.")
	dc.configMu.RLock()
	currentConfig := dc.config
	dc.configMu.RUnlock() // Read config safely.
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	reader, err := dc.client.GenerateStream(ctx, prompt, currentConfig)
	if err != nil {
		return "", err
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

// GetCompletionStreamFromFile provides context-aware completion using analysis, streaming the result.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, filename string, row, col int, w io.Writer) error {
	dc.configMu.RLock()
	currentConfig := dc.config
	dc.configMu.RUnlock() // Read config safely.
	var contextPreamble string = "// Basic file context only."
	var analysisInfo *AstContextInfo
	var analysisErr error

	if currentConfig.UseAst {
		log.Printf("Analyzing context (or checking cache) for %s:%d:%d", filename, row, col)
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second)
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, filename, row, col)
		cancelAnalysis()
		// Check only for *fatal* analysis errors here. Non-fatal are logged by Analyze.
		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) && !errors.Is(analysisErr, ErrCache) {
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		if analysisErr != nil {
			log.Printf("Non-fatal error during analysis/cache check for %s: %v", filename, analysisErr)
		} // Log non-fatal again for context.
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

	snippetCtx, err := extractSnippetContext(filename, row, col) // Use helper from utils.go
	if err != nil {
		return fmt.Errorf("failed to extract code snippet context: %w", err)
	} // Fatal if snippet fails.

	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	if len(prompt) > 1000 {
		log.Printf("Generated Prompt (length %d):\n---\n%s\n... (prompt truncated in log)\n---", len(prompt), prompt[:1000])
	} else {
		log.Printf("Generated Prompt (length %d):\n---\n%s\n---", len(prompt), prompt)
	}

	// Call Ollama API with retry logic.
	apiCallFunc := func() error {
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig)
		if apiErr != nil {
			var oe *OllamaError
			isRetryable := errors.As(apiErr, &oe) && (oe.Status == http.StatusServiceUnavailable || oe.Status == http.StatusTooManyRequests)
			isRetryable = isRetryable || errors.Is(apiErr, context.DeadlineExceeded)
			if isRetryable {
				return apiErr
			}
			return apiErr
		} // Return raw error if retryable, else already wrapped.
		streamErr := streamCompletion(apiCtx, reader, w) // Use helper from utils.go
		if streamErr != nil {
			if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
				return streamErr
			}
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		log.Println("Completion stream finished successfully for this attempt.")
		return nil
	}
	err = retry(ctx, apiCallFunc, maxRetries, retryDelay) // Use helper from utils.go
	if err != nil {
		var oe *OllamaError
		if errors.As(err, &oe) || errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			return err
		}
		return fmt.Errorf("failed to get completion stream after retries: %w", err)
	}

	if analysisErr != nil {
		log.Printf("Warning: Completion succeeded, but context analysis/cache check encountered non-fatal errors: %v", analysisErr)
	}
	return nil // Overall success.
}

// ============================================================================
// Internal Analysis & Preamble Helpers (Remain in main package file)
// ============================================================================
// These are kept here as they are tightly coupled with AstContextInfo and types.

// buildPreamble constructs the context string sent to the LLM from analysis info.
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
		} else {
			log.Printf("DEBUG: Preamble limit reached even before adding truncation marker for %s", section)
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
	formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker) // Pass qualifier here.
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

// formatEnclosingFuncSection formats the enclosing function/method info.
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
		return true
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
		fieldOrMethod, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
		isKnownMember := fieldOrMethod != nil
		unknownMemberMsg := ""
		if info.SelectorExprType != nil && !isKnownMember && selName != "" {
			unknownMemberMsg = fmt.Sprintf(" (unknown member '%s')", selName)
		}
		if !add(fmt.Sprintf("// Selector context: expr type = %s%s\n", typeName, unknownMemberMsg)) {
			return false
		}
		if info.SelectorExprType != nil && isKnownMember {
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
		} else if info.SelectorExprType != nil && !isKnownMember {
			if !add("//   (Cannot list members: selected member is unknown)\n") {
				return false
			}
		} else {
			if !add("//   (Cannot list members: type analysis failed for base expression)\n") {
				return false
			}
		}
		return true
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
		return true
	}
	return true
}

// formatScopeSection formats variables/constants/types in scope, respecting limits.
// Corrected signature: Now includes qualifier.
func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string)) bool {
	if len(info.VariablesInScope) == 0 {
		return true
	}
	if !add("// Variables/Constants/Types in Scope:\n") {
		return false
	}
	var items []string
	// Use the passed qualifier when formatting object strings.
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

// addAnalysisError adds a non-fatal error, avoiding duplicates based on message.
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

// logAnalysisErrors logs joined non-fatal analysis errors.
func logAnalysisErrors(errs []error) {
	if len(errs) > 0 {
		combinedErr := errors.Join(errs...)
		log.Printf("Context analysis completed with %d non-fatal error(s): %v", len(errs), combinedErr)
	}
}

// --- Other analysis helpers remain here as they are tightly coupled ---

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
					} else {
						posStr := ""
						if fset != nil {
							posStr = fset.Position(n.Pos()).String()
						}
						addAnalysisError(info, fmt.Errorf("scope info missing for block at %s", posStr))
					}
				} else if info.EnclosingBlock == n {
					reason := "type info unavailable"
					if targetPkg != nil && targetPkg.TypesInfo == nil {
						reason = "TypesInfo is nil"
					}
					if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes == nil {
						reason = "TypesInfo.Scopes is nil"
					}
					posStr := ""
					if fset != nil {
						posStr = fset.Position(n.Pos()).String()
					}
					addAnalysisError(info, fmt.Errorf("cannot get scope variables for block at %s: %s", posStr, reason))
				}
			}
		}
	}
	addPackageScope(targetPkg, info)
}

// addPackageScope adds package-level identifiers to the scope map.
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
		if v != nil && v.Name() != "" {
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
cleanup:
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
		} else if hasTypeInfo {
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
		} else if hasTypeInfo {
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
			} else if hasTypeInfo {
				addAnalysisError(info, errors.New("cannot analyze selector expr: Types map is nil"))
			}
			if info.SelectorExprType != nil && selExpr.Sel != nil {
				selName := selExpr.Sel.Name
				obj, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
				if obj == nil {
					addAnalysisError(info, fmt.Errorf("selecting unknown member '%s' from type '%s'", selName, info.SelectorExprType.String()))
				}
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
					addAnalysisError(info, fmt.Errorf("object not found for identifier '%s' at %s", ident.Name, posStr(ident.Pos())))
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
		}
		if cursorPos > argEnd {
			if i == len(args)-1 {
				return i + 1
			}
			if args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i + 1
			}
		}
	}
	if len(args) > 0 && args[0] != nil && cursorPos < args[0].Pos() {
		return 0
	}
	return 0
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
