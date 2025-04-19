// deepcomplete.go
// Package deepcomplete provides core logic for local code completion using LLMs.
// This version assumes helper functions have been moved to deepcomplete_helpers.go
package deepcomplete

import (
	"bytes"
	"context"
	"encoding/gob" // For cache serialization
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	"log/slog" // Cycle 3: Added slog
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/dgraph-io/ristretto" // Cycle 9: Added ristretto
	"go.etcd.io/bbolt"
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

	// Retry constants
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
		slog.Warn("max_tokens is not positive, using default", "configured_value", c.MaxTokens, "default", defaultMaxTokens)
		c.MaxTokens = defaultMaxTokens
	}
	if c.Temperature < 0 {
		slog.Warn("temperature is negative, using default", "configured_value", c.Temperature, "default", defaultTemperature)
		c.Temperature = defaultTemperature
	}
	if c.MaxPreambleLen <= 0 {
		slog.Warn("max_preamble_len is not positive, using default", "configured_value", c.MaxPreambleLen, "default", DefaultConfig.MaxPreambleLen)
		c.MaxPreambleLen = DefaultConfig.MaxPreambleLen
	}
	if c.MaxSnippetLen <= 0 {
		slog.Warn("max_snippet_len is not positive, using default", "configured_value", c.MaxSnippetLen, "default", DefaultConfig.MaxSnippetLen)
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
// Updated for Cycle 8/9/Hover.
type AstContextInfo struct {
	FilePath           string // Absolute, validated path
	Version            int    // Document version for memory cache keying
	CursorPos          token.Pos
	PackageName        string
	TargetPackage      *packages.Package // Store loaded package for qualifier/scope
	TargetFileSet      *token.FileSet    // Store FileSet for position info
	TargetAstFile      *ast.File         // Store AST for comment lookup? (Or use CommentMap)
	EnclosingFunc      *types.Func
	EnclosingFuncNode  *ast.FuncDecl
	ReceiverType       string
	EnclosingBlock     *ast.BlockStmt
	Imports            []*ast.ImportSpec
	CommentsNearCursor []string
	IdentifierAtCursor *ast.Ident
	IdentifierType     types.Type
	IdentifierObject   types.Object
	IdentifierDefNode  ast.Node // Added for Hover: Store the defining AST node
	SelectorExpr       *ast.SelectorExpr
	SelectorExprType   types.Type
	CallExpr           *ast.CallExpr
	CallExprFuncType   *types.Signature
	CallArgIndex       int
	ExpectedArgType    types.Type
	CompositeLit       *ast.CompositeLit
	CompositeLitType   types.Type
	VariablesInScope   map[string]types.Object
	PromptPreamble     string // Final preamble generated (potentially cached)
	AnalysisErrors     []error
	// Potentially add CommentMap here if needed for hover doc lookup
	// CommentMap         ast.CommentMap
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

// CachedAnalysisData holds derived information stored in the bbolt cache (gob-encoded).
type CachedAnalysisData struct {
	PackageName    string
	PromptPreamble string
	// Add other easily serializable, frequently reused data if needed
}

// CachedAnalysisEntry represents the full structure stored in bbolt.
type CachedAnalysisEntry struct {
	SchemaVersion   int
	GoModHash       string
	InputFileHashes map[string]string // key: relative path (using '/')
	AnalysisGob     []byte            // Gob-encoded CachedAnalysisData
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
	ErrAnalysisFailed       = errors.New("code analysis failed") // Wraps non-fatal analysis issues
	ErrOllamaUnavailable    = errors.New("ollama API unavailable")
	ErrStreamProcessing     = errors.New("error processing LLM stream")
	ErrConfig               = errors.New("configuration error") // Wraps non-fatal config load issues
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
	ErrInvalidURI           = errors.New("invalid document URI") // Added for path validation
)

// =============================================================================
// Interfaces for Components (Remain in main package file)
// =============================================================================

// LLMClient defines interaction with the LLM API.
type LLMClient interface {
	GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error)
}

// Analyzer defines code context analysis.
// Updated for Cycle 9 versioning and fixed DocumentURI usage.
type Analyzer interface {
	// Analyze performs code analysis for a given file and position.
	// filename should be an absolute, validated path.
	// version is the document version from the client.
	Analyze(ctx context.Context, filename string, version int, line, col int) (*AstContextInfo, error)
	Close() error
	// InvalidateCache invalidates the bbolt cache for a directory.
	InvalidateCache(dir string) error
	// InvalidateMemoryCacheForURI invalidates the ristretto cache for a URI.
	// **FIX:** Changed DocumentURI to string to avoid cross-package dependency.
	InvalidateMemoryCacheForURI(uri string, version int) error
	// MemoryCacheEnabled checks if the ristretto cache is active.
	MemoryCacheEnabled() bool
	// GetMemoryCacheMetrics returns the ristretto cache metrics.
	GetMemoryCacheMetrics() *ristretto.Metrics
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
	// Ensure default slog logger is set for potential warnings here
	// (Assuming it's set in main or tests)
	cfg := DefaultConfig
	var loadedFromFile bool
	var loadErrors []error
	var configParseError error

	primaryPath, secondaryPath, pathErr := getConfigPaths()
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		slog.Warn("Could not determine config paths", "error", pathErr)
	}

	// Try loading from primary path
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
			slog.Info("Loaded config", "path", primaryPath)
		}
	}

	// Try secondary path if primary wasn't found or didn't load successfully
	primaryNotFoundOrFailed := !loadedFromFile || len(loadErrors) > 0
	if primaryNotFoundOrFailed && secondaryPath != "" {
		loaded, loadErr := loadAndMergeConfig(secondaryPath, &cfg)
		if loadErr != nil {
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				if configParseError == nil {
					configParseError = loadErr
				} // Keep first parse error
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", secondaryPath, loadErr))
		}
		// Only set loadedFromFile if it wasn't already true from primary
		if !loadedFromFile {
			loadedFromFile = loaded
		}
		if loaded && loadErr == nil {
			slog.Info("Loaded config", "path", secondaryPath)
		}
	}

	// Write default config if no file was loaded successfully
	loadSucceeded := loadedFromFile && configParseError == nil
	if !loadSucceeded {
		if configParseError != nil {
			slog.Warn("Existing config file failed to parse. Attempting to write default.", "error", configParseError)
		} else {
			slog.Info("No valid config file found. Attempting to write default.")
		}
		// Determine write path (prefer primary)
		writePath := primaryPath
		if writePath == "" {
			writePath = secondaryPath
		}

		if writePath != "" {
			slog.Info("Attempting to write default config", "path", writePath)
			if err := writeDefaultConfig(writePath, DefaultConfig); err != nil {
				slog.Warn("Failed to write default config", "error", err)
				loadErrors = append(loadErrors, fmt.Errorf("writing default config failed: %w", err))
			}
		} else {
			slog.Warn("Cannot determine path to write default config.")
			loadErrors = append(loadErrors, errors.New("cannot determine default config path"))
		}
		cfg = DefaultConfig // Use defaults if write fails or no path
	}

	// Ensure internal templates are set
	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	// Final validation of the resulting config
	finalCfg := cfg
	if err := finalCfg.Validate(); err != nil {
		slog.Warn("Config after load/merge failed validation. Returning pure defaults.", "error", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		// Validate pure defaults as a safety check
		if valErr := DefaultConfig.Validate(); valErr != nil {
			slog.Error("FATAL: Default config is invalid", "error", valErr)
			return DefaultConfig, fmt.Errorf("default config is invalid: %w", valErr)
		}
		finalCfg = DefaultConfig
	}

	// Return config and potentially wrapped non-fatal errors
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
		slog.Warn("Could not determine user config directory", "error", cfgErr)
	}
	homeDir, homeErr := os.UserHomeDir()
	if homeErr == nil {
		secondary = filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
		// If primary failed, use secondary as primary (common on some systems)
		if primary == "" && cfgErr != nil {
			primary = secondary
			slog.Debug("Using fallback primary config path", "path", primary)
			secondary = "" // No need for secondary if it's the same as primary fallback
		}
		// Avoid listing same path twice
		if primary == secondary {
			secondary = ""
		}
	} else {
		slog.Warn("Could not determine user home directory", "error", homeErr)
	}
	// If neither path could be determined, return an error
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
		} // File not found is not an error here
		return false, fmt.Errorf("reading config file %q failed: %w", path, err)
	}
	if len(data) == 0 {
		slog.Warn("Config file exists but is empty, ignoring.", "path", path)
		return true, nil // Treat as loaded but empty
	}

	var fileCfg FileConfig
	if err := json.Unmarshal(data, &fileCfg); err != nil {
		return true, fmt.Errorf("parsing config file JSON %q failed: %w", path, err)
	}

	// Merge loaded fields into cfg, overwriting defaults
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

	return true, nil // Loaded successfully
}

// writeDefaultConfig creates the directory and writes the default config as JSON.
func writeDefaultConfig(path string, defaultConfig Config) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0750); err != nil { // Use 0750 for permissions
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}
	// Create an exportable struct containing only the fields to write
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
		OllamaURL: defaultConfig.OllamaURL, Model: defaultConfig.Model, MaxTokens: defaultConfig.MaxTokens,
		Stop: defaultConfig.Stop, Temperature: defaultConfig.Temperature, UseAst: defaultConfig.UseAst,
		UseFim: defaultConfig.UseFim, MaxPreambleLen: defaultConfig.MaxPreambleLen, MaxSnippetLen: defaultConfig.MaxSnippetLen,
	}
	jsonData, err := json.MarshalIndent(expCfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal default config to JSON: %w", err)
	}
	// Use more restrictive permissions for the config file itself
	if err := os.WriteFile(path, jsonData, 0640); err != nil {
		return fmt.Errorf("failed to write default config file %s: %w", path, err)
	}
	slog.Info("Wrote default configuration", "path", path)
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
	// Configure HTTP client with appropriate timeouts
	return &httpOllamaClient{
		httpClient: &http.Client{
			Timeout: 90 * time.Second, // Overall request timeout
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: 10 * time.Second, // Connection timeout
				}).DialContext,
				TLSHandshakeTimeout: 10 * time.Second, // TLS handshake timeout
				// Add other transport settings if needed (e.g., proxy)
			},
		},
	}
}

// GenerateStream sends a request to Ollama's /api/generate endpoint.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	base := strings.TrimSuffix(config.OllamaURL, "/")
	endpointURL := base + "/api/generate"
	u, err := url.Parse(endpointURL)
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL '%s': %w", endpointURL, err)
	}

	// Construct payload
	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"temperature": config.Temperature,
			"num_ctx":     4096, // Consider making configurable or deriving
			"top_p":       0.9,  // Common default, consider making configurable
			"stop":        config.Stop,
			"num_predict": config.MaxTokens,
		},
	}
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON payload: %w", err)
	}

	// Create request with context
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/x-ndjson") // Expect newline-delimited JSON stream

	// Execute request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Check for context cancellation first
		if errors.Is(err, context.Canceled) {
			return nil, context.Canceled // Propagate cancellation
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		// Check for network errors (e.g., connection refused)
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			// Network timeout (e.g., connection timeout)
			return nil, fmt.Errorf("%w: network timeout connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		if opErr, ok := err.(*net.OpError); ok && opErr.Op == "dial" {
			return nil, fmt.Errorf("%w: connection refused or network error connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		// Other HTTP errors
		return nil, fmt.Errorf("%w: http request failed: %w", ErrOllamaUnavailable, err)
	}

	// Check response status code
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(failed to read error response body)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			// Try to parse Ollama's specific error format
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error
			}
		}
		// Create specific OllamaError
		err = &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
	}

	// Return response body for streaming
	return resp.Body, nil
}

// GoPackagesAnalyzer implements Analyzer using go/packages and caching.
// Updated for Cycle 9 Ristretto cache.
type GoPackagesAnalyzer struct {
	db          *bbolt.DB        // Disk cache
	memoryCache *ristretto.Cache // In-memory cache
	mu          sync.Mutex       // Protects access to db handle during Close.
}

// NewGoPackagesAnalyzer initializes the analyzer and caches.
func NewGoPackagesAnalyzer() *GoPackagesAnalyzer {
	// --- bbolt Cache Setup ---
	dbPath := ""
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil {
			dbPath = filepath.Join(dbDir, "analysis_cache.db")
		} else {
			slog.Warn("Could not create bbolt cache directory", "path", dbDir, "error", err)
		}
	} else {
		slog.Warn("Could not determine user cache directory. Bbolt caching disabled.", "error", err)
	}

	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second}
		db, err = bbolt.Open(dbPath, 0600, opts)
		if err != nil {
			slog.Warn("Failed to open bbolt cache file. Bbolt caching will be disabled.", "path", dbPath, "error", err)
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
				slog.Warn("Failed to ensure bbolt bucket exists. Bbolt caching disabled.", "error", err)
				db.Close()
				db = nil
			} else {
				slog.Info("Using bbolt cache", "path", dbPath, "schema", cacheSchemaVersion)
			}
		}
	}

	// --- Ristretto Cache Setup (Cycle 9) ---
	memCache, cacheErr := ristretto.NewCache(&ristretto.Config{
		NumCounters: 1e7,     // 10M keys to track frequency. Tune based on usage.
		MaxCost:     1 << 30, // 1GB max cache size. Tune based on memory.
		BufferItems: 64,      // Default is fine.
		Metrics:     true,    // Enable metrics collection.
	})
	if cacheErr != nil {
		slog.Warn("Failed to create ristretto memory cache. In-memory caching disabled.", "error", cacheErr)
		memCache = nil // Ensure it's nil if setup fails
	} else {
		slog.Info("Initialized ristretto in-memory cache", "max_cost", "1GB")
	}

	return &GoPackagesAnalyzer{db: db, memoryCache: memCache}
}

// Close closes cache connections.
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	var closeErrors []error
	if a.db != nil {
		slog.Info("Closing bbolt cache database.")
		if err := a.db.Close(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("bbolt close failed: %w", err))
		}
		a.db = nil
	}
	if a.memoryCache != nil {
		slog.Info("Closing ristretto memory cache.")
		a.memoryCache.Close() // Waits for buffers etc.
		a.memoryCache = nil
	}
	if len(closeErrors) > 0 {
		return errors.Join(closeErrors...)
	}
	return nil
}

// Analyze performs code analysis, orchestrating calls to helpers.
// Updated for Cycle 8/9/Hover/Defensive.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, absFilename string, version int, line, col int) (info *AstContextInfo, analysisErr error) {
	// Input filename should already be validated absolute path by caller
	logger := slog.Default().With("absFile", absFilename, "version", version, "line", line, "col", col)

	// Initialize result struct
	info = &AstContextInfo{
		FilePath:         absFilename,
		Version:          version,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		CallArgIndex:     -1, // Initialize to -1
	}

	// Panic recovery for the entire analysis process
	defer func() {
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			logger.Error("Panic recovered during Analyze", "error", r, "stack", string(debug.Stack()))
			addAnalysisError(info, panicErr, logger)
			// Ensure analysisErr reflects the panic if no other error was set
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr)
			}
		}
		// Aggregate non-fatal errors into the final returned error
		if len(info.AnalysisErrors) > 0 {
			finalErr := errors.Join(info.AnalysisErrors...)
			// Wrap non-fatal errors in ErrAnalysisFailed
			if analysisErr == nil {
				analysisErr = fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr)
			} else {
				analysisErr = fmt.Errorf("%w: %w", analysisErr, finalErr)
			} // Join with existing fatal error if any
		}
	}()

	logger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)

	// --- Bbolt Cache Check (Disk Cache for final preamble) ---
	goModHash := calculateGoModHash(dir) // Uses slog internally now
	cacheKey := []byte(dir + "::" + goModHash)
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry
	var loadDuration, stepsDuration, preambleDuration time.Duration // Timings

	if a.db != nil {
		// ... (bbolt cache read logic as before, using logger, calling deleteCacheEntryByKey on error) ...
		readStart := time.Now()
		dbErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				logger.Debug("Bbolt cache bucket not found.")
				return nil
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				logger.Debug("Bbolt cache miss.", "key", string(cacheKey))
				return nil
			}
			logger.Debug("Bbolt cache hit (raw bytes found). Decoding...", "key", string(cacheKey))
			var decoded CachedAnalysisEntry
			decoder := gob.NewDecoder(bytes.NewReader(valBytes))
			if err := decoder.Decode(&decoded); err != nil {
				return fmt.Errorf("%w: %w", ErrCacheDecode, err)
			} // Return error to trigger delete below
			if decoded.SchemaVersion != cacheSchemaVersion {
				logger.Warn("Cache data has old schema version. Ignoring.", "key", string(cacheKey))
				return nil
			} // Treat as miss
			cachedEntry = &decoded
			return nil
		})
		if dbErr != nil {
			logger.Warn("Error reading or decoding from bbolt cache. Cache check failed.", "error", dbErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbErr), logger)
			if errors.Is(dbErr, ErrCacheDecode) {
				go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "decode_failure"))
			}
			cachedEntry = nil
		} else if cachedEntry != nil && cachedEntry.SchemaVersion != cacheSchemaVersion {
			go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "schema_mismatch"))
			cachedEntry = nil
		}
		logger.Debug("Bbolt cache read attempt finished", "duration", time.Since(readStart))

		// Validate cache hit based on file hashes.
		if cachedEntry != nil {
			validationStart := time.Now()
			logger.Debug("Potential bbolt cache hit. Validating file hashes...", "key", string(cacheKey))
			currentHashes, hashErr := calculateInputHashes(dir, nil) // Pass nil pkg here.
			if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes) {
				logger.Debug("Bbolt cache VALID. Attempting to decode analysis data...", "key", string(cacheKey))
				decodeStart := time.Now()
				var analysisData CachedAnalysisData
				decoder := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob))
				if decodeErr := decoder.Decode(&analysisData); decodeErr == nil {
					info.PackageName = analysisData.PackageName
					info.PromptPreamble = analysisData.PromptPreamble
					cacheHit = true
					loadDuration = time.Since(decodeStart)
					logger.Debug("Analysis data successfully decoded from bbolt cache.", "duration", loadDuration)
					logger.Debug("Using cached preamble. Skipping packages.Load and analysis steps.", "preamble_length", len(info.PromptPreamble))
				} else {
					logger.Warn("Failed to gob-decode cached analysis data. Treating as miss.", "error", decodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheDecode, decodeErr), logger)
					go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "analysis_decode_failure"))
					cacheHit = false
				}
			} else {
				logger.Debug("Bbolt cache INVALID (hash mismatch or error). Treating as miss.", "key", string(cacheKey), "hash_error", hashErr)
				go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "hash_mismatch"))
				if hashErr != nil {
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger)
				}
				cacheHit = false
			}
			logger.Debug("Bbolt cache validation/decode finished", "duration", time.Since(validationStart))
		}
	} else {
		logger.Debug("Bbolt cache disabled (db handle is nil).")
	}

	// --- Perform Full Analysis if Cache Miss ---
	if !cacheHit {
		logger.Debug("Bbolt cache miss or invalid. Performing full analysis...", "key", string(cacheKey))

		// --- Step 1: Load Package Info ---
		loadStart := time.Now()
		fset := token.NewFileSet() // Create new FileSet for this analysis run
		info.TargetFileSet = fset  // Store FileSet in info for later use (e.g., position formatting)
		// **FIX:** Call the function now defined in helpers
		targetPkg, targetFileAST, targetFile, loadErrors := loadPackageAndFile(ctx, absFilename, fset, logger)
		loadDuration = time.Since(loadStart)
		logger.Debug("packages.Load completed", "duration", loadDuration)
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr, logger)
		}
		info.TargetPackage = targetPkg     // Store package info
		info.TargetAstFile = targetFileAST // Store AST

		// --- Step 2: Perform Detailed Analysis Steps ---
		stepsStart := time.Now()
		if targetFile != nil { // Proceed only if the target token.File was found
			// Call refactored helper (Cycle 8), passing analyzer for memory cache (Cycle 9)
			analyzeStepErr := performAnalysisSteps(targetFile, targetFileAST, targetPkg, fset, line, col, a, info, logger)
			if analyzeStepErr != nil {
				addAnalysisError(info, analyzeStepErr, logger)
			}
		} else {
			// Log error if targetFile is nil but loading didn't report critical error earlier
			if len(loadErrors) == 0 {
				addAnalysisError(info, errors.New("cannot perform analysis steps: target token.File is nil"), logger)
			}
			// Attempt to gather basic package scope even without file context
			gatherScopeContext(nil, targetPkg, fset, info, logger)
		}
		stepsDuration = time.Since(stepsStart)
		logger.Debug("Analysis steps completed", "duration", stepsDuration)

		// --- Step 3: Build Preamble ---
		preambleStart := time.Now()
		var qualifier types.Qualifier
		if targetPkg != nil && targetPkg.Types != nil {
			if info.PackageName == "" {
				info.PackageName = targetPkg.Types.Name()
			}
			qualifier = types.RelativeTo(targetPkg.Types)
		} else {
			qualifier = func(other *types.Package) string {
				if other != nil {
					return other.Path()
				}
				return ""
			}
			logger.Debug("Building preamble with limited/no type info.")
		}
		// Call refactored helper (Cycle 8), passing analyzer for potential future caching (Cycle 9)
		info.PromptPreamble = constructPromptPreamble(a, info, qualifier, logger)
		preambleDuration = time.Since(preambleStart)
		logger.Debug("Preamble construction completed", "duration", preambleDuration)

		// --- Step 4: Save to Bbolt Cache ---
		// Save only if analysis didn't have critical load errors and preamble was generated
		shouldSave := a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0
		if shouldSave {
			// ... (bbolt cache save logic as before, using logger) ...
			logger.Debug("Attempting to save analysis results to bbolt cache.", "key", string(cacheKey))
			saveStart := time.Now()
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg)
			if hashErr == nil {
				analysisDataToCache := CachedAnalysisData{PackageName: info.PackageName, PromptPreamble: info.PromptPreamble}
				var gobBuf bytes.Buffer
				if encodeErr := gob.NewEncoder(&gobBuf).Encode(&analysisDataToCache); encodeErr == nil {
					analysisGob := gobBuf.Bytes()
					entryToSave := CachedAnalysisEntry{SchemaVersion: cacheSchemaVersion, GoModHash: goModHash, InputFileHashes: inputHashes, AnalysisGob: analysisGob}
					var entryBuf bytes.Buffer
					if entryEncodeErr := gob.NewEncoder(&entryBuf).Encode(&entryToSave); entryEncodeErr == nil {
						encodedBytes := entryBuf.Bytes()
						saveErr := a.db.Update(func(tx *bbolt.Tx) error {
							b := tx.Bucket(cacheBucketName)
							if b == nil {
								return fmt.Errorf("%w: cache bucket %s disappeared", ErrCacheWrite, string(cacheBucketName))
							}
							logger.Debug("Writing bytes to bbolt cache", "key", string(cacheKey), "bytes", len(encodedBytes))
							return b.Put(cacheKey, encodedBytes)
						})
						if saveErr == nil {
							logger.Debug("Saved analysis results to bbolt cache", "key", string(cacheKey), "duration", time.Since(saveStart))
						} else {
							logger.Warn("Failed to write to bbolt cache", "key", string(cacheKey), "error", saveErr)
							addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheWrite, saveErr), logger)
						}
					} else { /* handle entry encode error */
						addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr), logger)
					}
				} else { /* handle data encode error */
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr), logger)
				}
			} else { /* handle hash error */
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger)
			}
		} else if a.db != nil {
			logger.Debug("Skipping bbolt cache save", "key", string(cacheKey), "load_errors", len(loadErrors), "preamble_empty", info.PromptPreamble == "")
		}
	}

	// Log final timing summary
	if cacheHit {
		logger.Info("Context analysis finished (bbolt cache hit)", "decode_duration", loadDuration)
	} else {
		logger.Info("Context analysis finished (full analysis)", "load_duration", loadDuration, "steps_duration", stepsDuration, "preamble_duration", preambleDuration)
	}
	logger.Debug("Final Context Preamble generated", "length", len(info.PromptPreamble))

	// Return info and potentially wrapped non-fatal errors (handled by defer)
	return info, analysisErr
}

// InvalidateCache removes the bbolt cached entry for a given directory.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	logger := slog.Default().With("dir", dir)
	a.mu.Lock()
	db := a.db // Access db safely
	a.mu.Unlock()
	if db == nil {
		logger.Debug("Bbolt cache invalidation skipped: DB is nil.")
		return nil
	}
	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	logger.Info("Invalidating bbolt cache entry", "key", string(cacheKey))
	// Pass logger to helper
	return deleteCacheEntryByKey(db, cacheKey, logger)
}

// InvalidateMemoryCacheForURI clears relevant entries from the ristretto cache.
// Cycle 9: Added method.
// **FIX:** Changed DocumentURI to string.
func (a *GoPackagesAnalyzer) InvalidateMemoryCacheForURI(uri string, version int) error {
	logger := slog.Default().With("uri", uri, "version", version)
	a.mu.Lock()
	memCache := a.memoryCache // Access cache safely
	a.mu.Unlock()

	if memCache == nil {
		logger.Debug("Memory cache invalidation skipped: Cache is nil.")
		return nil
	}

	logger.Info("Invalidating memory cache (placeholder logic)")
	// --- Placeholder Invalidation Logic ---
	// This is highly dependent on the key structure chosen in Cycle 9, Step 3.
	// Option 1: Clear the entire cache (simple, blunt)
	// memCache.Clear()
	// logger.Warn("Cleared entire memory cache due to document change (inefficient).")

	// Option 2: Delete keys based on prefix (requires specific key design)
	// Example: If keys are "scope:<uri>:<version>:...", "preamble:<uri>:<version>:..."
	// We would need a way to find/delete keys matching "scope:<uri>:<version>".
	// Ristretto doesn't directly support prefix deletion. This might involve:
	//   a) Keeping a separate index (e.g., map[string][]cacheKey).
	//   b) Iterating through *all* keys (if possible via metrics/internal access - not standard API).
	//   c) Using a different cache library with prefix support.

	// For now, log that invalidation is needed but not fully implemented.
	logger.Warn("Memory cache invalidation logic is currently a placeholder and may not remove all stale entries.")
	// --- End Placeholder ---

	return nil
}

// MemoryCacheEnabled checks if the ristretto cache is active. (Cycle 9)
func (a *GoPackagesAnalyzer) MemoryCacheEnabled() bool {
	a.mu.Lock() // Although reading a pointer might be atomic, lock for consistency
	defer a.mu.Unlock()
	return a.memoryCache != nil
}

// GetMemoryCacheMetrics returns the ristretto cache metrics. (Cycle 9)
// Returns nil if cache is disabled.
func (a *GoPackagesAnalyzer) GetMemoryCacheMetrics() *ristretto.Metrics {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.memoryCache != nil {
		return a.memoryCache.Metrics
	}
	return nil
}

// --- Prompt Formatter ---
type templateFormatter struct{}

func newTemplateFormatter() *templateFormatter { return &templateFormatter{} }

// FormatPrompt combines context and snippet into the final LLM prompt, applying truncation.
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate // Use standard template by default
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2 // Divide budget for FIM prefix/suffix

	// Truncate context preamble if necessary
	if len(contextPreamble) > maxPreambleLen {
		slog.Warn("Truncating context preamble", "original_length", len(contextPreamble), "max_length", maxPreambleLen)
		marker := "... (context truncated)\n"
		startByte := len(contextPreamble) - maxPreambleLen + len(marker)
		if startByte < 0 {
			startByte = 0
		} // Avoid negative index
		contextPreamble = marker + contextPreamble[startByte:]
	}

	// Handle FIM (Fill-In-the-Middle) vs standard completion
	if config.UseFim {
		template = config.FimTemplate // Use FIM template
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix

		// Truncate FIM prefix if necessary (from the beginning)
		if len(prefix) > maxFIMPartLen {
			slog.Warn("Truncating FIM prefix", "original_length", len(prefix), "max_length", maxFIMPartLen)
			marker := "...(prefix truncated)"
			startByte := len(prefix) - maxFIMPartLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			prefix = marker + prefix[startByte:]
		}
		// Truncate FIM suffix if necessary (from the end)
		if len(suffix) > maxFIMPartLen {
			slog.Warn("Truncating FIM suffix", "original_length", len(suffix), "max_length", maxFIMPartLen)
			marker := "(suffix truncated)..."
			endByte := maxFIMPartLen - len(marker)
			if endByte < 0 {
				endByte = 0
			}
			suffix = suffix[:endByte] + marker
		}
		finalPrompt = fmt.Sprintf(template, contextPreamble, prefix, suffix)
	} else {
		// Standard completion: use only the prefix as the snippet
		snippet := snippetCtx.Prefix
		if len(snippet) > maxSnippetLen {
			slog.Warn("Truncating code snippet (prefix)", "original_length", len(snippet), "max_length", maxSnippetLen)
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
	analyzer  Analyzer // Use the interface type
	formatter PromptFormatter
	config    Config
	configMu  sync.RWMutex // Protects concurrent read/write access to config.
}

// NewDeepCompleter creates a new DeepCompleter service with default config.
func NewDeepCompleter() (*DeepCompleter, error) {
	cfg, configErr := LoadConfig()
	// Log warning but continue if only ErrConfig occurred
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		return nil, configErr // Fatal error during config load
	}
	if configErr != nil {
		slog.Warn("Warning during initial config load", "error", configErr)
	}

	// Validate the loaded/default config before creating components
	if err := cfg.Validate(); err != nil {
		slog.Warn("Initial config is invalid after load/merge. Using pure defaults.", "error", err)
		cfg = DefaultConfig // Reset to pure defaults
		if valErr := cfg.Validate(); valErr != nil {
			slog.Error("Default config validation failed", "error", valErr)
			return nil, fmt.Errorf("default config validation failed: %w", valErr)
		}
	}

	// Initialize components
	analyzer := NewGoPackagesAnalyzer() // Initializes bbolt and ristretto
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer, // Store concrete type satisfying interface
		formatter: newTemplateFormatter(),
		config:    cfg,
	}

	// Return the completer, potentially along with the non-fatal config load warning
	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with provided config.
func NewDeepCompleterWithConfig(config Config) (*DeepCompleter, error) {
	// Ensure templates are set if empty
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	// Validate the provided config
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	// Initialize components
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
		return dc.analyzer.Close() // Close handles both bbolt and ristretto
	}
	return nil
}

// UpdateConfig atomically updates the completer's configuration after validation.
func (dc *DeepCompleter) UpdateConfig(newConfig Config) error {
	// Ensure templates are set if empty
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate
	}
	// Validate before applying
	if err := newConfig.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	// Apply atomically
	dc.configMu.Lock()
	defer dc.configMu.Unlock()
	dc.config = newConfig
	// Use structured logging for config values
	slog.Info("DeepCompleter configuration updated",
		slog.String("ollama_url", newConfig.OllamaURL),
		slog.String("model", newConfig.Model),
		slog.Int("max_tokens", newConfig.MaxTokens),
		slog.Any("stop", newConfig.Stop),
		slog.Float64("temperature", newConfig.Temperature),
		slog.Bool("use_ast", newConfig.UseAst),
		slog.Bool("use_fim", newConfig.UseFim),
		slog.Int("max_preamble_len", newConfig.MaxPreambleLen),
		slog.Int("max_snippet_len", newConfig.MaxSnippetLen),
	)
	return nil
}

// GetCurrentConfig returns a copy of the current configuration safely.
func (dc *DeepCompleter) GetCurrentConfig() Config {
	dc.configMu.RLock()
	defer dc.configMu.RUnlock()
	// Return a copy to prevent external modification
	cfgCopy := dc.config
	if cfgCopy.Stop != nil {
		stopsCopy := make([]string, len(cfgCopy.Stop))
		copy(stopsCopy, cfgCopy.Stop)
		cfgCopy.Stop = stopsCopy
	}
	return cfgCopy
}

// InvalidateAnalyzerCache provides access to the analyzer's bbolt invalidation logic.
func (dc *DeepCompleter) InvalidateAnalyzerCache(dir string) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	return dc.analyzer.InvalidateCache(dir)
}

// InvalidateMemoryCacheForURI provides access to the analyzer's memory cache invalidation.
// **FIX:** Changed DocumentURI to string.
func (dc *DeepCompleter) InvalidateMemoryCacheForURI(uri string, version int) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	return dc.analyzer.InvalidateMemoryCacheForURI(uri, version)
}

// GetAnalyzer returns the analyzer instance (needed for Cycle 9 metrics/invalidation).
// Consider if this is the best way to expose cache access.
func (dc *DeepCompleter) GetAnalyzer() Analyzer {
	return dc.analyzer
}

// GetCompletion provides basic completion for a direct code snippet (no AST analysis).
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	logger := slog.Default().With("operation", "GetCompletion")
	logger.Info("Handling basic completion request")

	currentConfig := dc.GetCurrentConfig() // Get config safely

	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	logger.Debug("Generated basic prompt", "length", len(prompt))

	// Call the LLM client via retry logic
	var buffer bytes.Buffer
	apiCallFunc := func() error {
		select {
		case <-ctx.Done():
			return ctx.Err() // Check context before attempt
		default:
		}
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()

		logger.Debug("Calling Ollama GenerateStream for basic completion")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig)
		if apiErr != nil {
			return apiErr
		} // Return error to potentially retry

		// Process stream
		streamCtx, cancelStream := context.WithTimeout(apiCtx, 50*time.Second)
		defer cancelStream()
		buffer.Reset()                                            // Clear buffer for this attempt
		streamErr := streamCompletion(streamCtx, reader, &buffer) // Uses slog
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		} // Wrap stream errors
		return nil // Success for this attempt
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // retry uses slog
	if err != nil {
		// Check context cancellation one last time
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
		// Categorize final error
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			logger.Error("Ollama unavailable for basic completion after retries", "error", err)
			return "", fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error for basic completion after retries", "error", err)
			return "", err // Return the wrapped ErrStreamProcessing
		}
		logger.Error("Failed to get basic completion after retries", "error", err)
		return "", fmt.Errorf("failed to get basic completion after %d retries: %w", maxRetries, err)
	}

	logger.Info("Basic completion successful")
	return strings.TrimSpace(buffer.String()), nil
}

// GetCompletionStreamFromFile provides context-aware completion using analysis, streaming the result.
// Updated for Cycle 9 versioning and expects validated filename.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, absFilename string, version int, line, col int, w io.Writer) error {
	// Assume absFilename is already validated by the caller (handler)
	logger := slog.Default().With("operation", "GetCompletionStreamFromFile", "path", absFilename, "version", version, "line", line, "col", col)

	currentConfig := dc.GetCurrentConfig() // Get config safely
	var contextPreamble string = "// Basic file context only."
	var analysisInfo *AstContextInfo
	var analysisErr error // To store non-fatal analysis errors

	// Perform analysis if enabled
	if currentConfig.UseAst {
		logger.Info("Analyzing context (or checking cache)")
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second)
		// Pass version to Analyze
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, absFilename, version, line, col)
		cancelAnalysis()

		// Handle analysis errors (fatal vs non-fatal)
		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			logger.Error("Fatal error during analysis/cache check", "error", analysisErr)
			return fmt.Errorf("analysis failed fatally: %w", analysisErr) // Return fatal errors immediately
		}
		// Keep non-fatal errors (wrapped in ErrAnalysisFailed) to potentially notify user later
		if analysisErr != nil {
			logger.Warn("Non-fatal error during analysis/cache check", "error", analysisErr)
		}

		// Use preamble if available
		if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else if analysisErr != nil {
			contextPreamble += fmt.Sprintf("\n// Warning: Context analysis completed with errors: %v\n", analysisErr)
		} else {
			contextPreamble += "\n// Warning: Context analysis returned no specific context preamble.\n"
		}
	} else {
		logger.Info("AST analysis disabled by config.")
	}

	// Extract code snippet around cursor
	// Pass validated absFilename
	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col) // Uses slog
	if snippetErr != nil {
		logger.Error("Failed to extract code snippet context", "error", snippetErr)
		return fmt.Errorf("failed to extract code snippet context: %w", snippetErr)
	}

	// Format the final prompt
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig) // Uses slog
	logger.Debug("Generated prompt", "length", len(prompt))

	// --- API Call with Retry ---
	apiCallFunc := func() error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()

		logger.Debug("Calling Ollama GenerateStream")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig)
		if apiErr != nil {
			return apiErr
		} // Let retry handler classify

		// Process stream, write directly to provided writer w
		streamErr := streamCompletion(apiCtx, reader, w) // streamCompletion uses slog
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		} // Wrap stream errors
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // retry uses slog
	if err != nil {
		select {
		case <-ctx.Done():
			return ctx.Err() // Prefer context error if cancelled
		default:
		}
		// Categorize final error
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			logger.Error("Ollama unavailable for stream after retries", "error", err)
			// Return specific error for handler to potentially notify user
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error for stream after retries", "error", err)
			return err // Return wrapped ErrStreamProcessing
		}
		logger.Error("Failed to get completion stream after retries", "error", err)
		return fmt.Errorf("failed to get completion stream after %d retries: %w", maxRetries, err)
	}

	// Log analysis warnings if stream succeeded
	if analysisErr != nil {
		logger.Warn("Completion stream successful, but context analysis encountered non-fatal errors", "analysis_error", analysisErr)
		// Don't return analysisErr here, as completion succeeded. Handler might notify user based on analysisErr.
	}
	logger.Info("Completion stream successful")
	return nil // Success
}
