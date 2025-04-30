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
	"go/token"
	"go/types"
	"io"
	"log/slog" // Use structured logging
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

	defaultMaxTokens      = 256            // Default maximum tokens for LLM response.
	DefaultStop           = "\n"           // Default stop sequence for LLM. Exported for CLI use.
	defaultTemperature    = 0.1            // Default sampling temperature for LLM.
	defaultLogLevel       = "info"         // Default log level.
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
	LogLevel       string   `json:"log_level"`        // Log level (debug, info, warn, error).
	UseAst         bool     `json:"use_ast"`          // Enable AST/Type analysis.
	UseFim         bool     `json:"use_fim"`          // Use Fill-in-the-Middle prompting.
	MaxPreambleLen int      `json:"max_preamble_len"` // Max bytes for AST context preamble.
	MaxSnippetLen  int      `json:"max_snippet_len"`  // Max bytes for code snippet context.
}

// Validate checks if configuration values are valid, applying defaults for some fields.
func (c *Config) Validate() error {
	var validationErrors []error
	logger := slog.Default() // Use default logger for validation warnings

	if strings.TrimSpace(c.OllamaURL) == "" {
		validationErrors = append(validationErrors, errors.New("ollama_url cannot be empty"))
	} else if _, err := url.ParseRequestURI(c.OllamaURL); err != nil {
		validationErrors = append(validationErrors, fmt.Errorf("invalid ollama_url: %w", err))
	}
	if strings.TrimSpace(c.Model) == "" {
		validationErrors = append(validationErrors, errors.New("model cannot be empty"))
	}
	if c.MaxTokens <= 0 {
		logger.Warn("Config validation: max_tokens is not positive, applying default.", "configured_value", c.MaxTokens, "default", DefaultConfig.MaxTokens)
		c.MaxTokens = DefaultConfig.MaxTokens
	}
	if c.Temperature < 0 {
		logger.Warn("Config validation: temperature is negative, applying default.", "configured_value", c.Temperature, "default", DefaultConfig.Temperature)
		c.Temperature = DefaultConfig.Temperature
	}
	if c.MaxPreambleLen <= 0 {
		logger.Warn("Config validation: max_preamble_len is not positive, applying default.", "configured_value", c.MaxPreambleLen, "default", DefaultConfig.MaxPreambleLen)
		c.MaxPreambleLen = DefaultConfig.MaxPreambleLen
	}
	if c.MaxSnippetLen <= 0 {
		logger.Warn("Config validation: max_snippet_len is not positive, applying default.", "configured_value", c.MaxSnippetLen, "default", DefaultConfig.MaxSnippetLen)
		c.MaxSnippetLen = DefaultConfig.MaxSnippetLen
	}
	if c.LogLevel == "" {
		logger.Warn("Config validation: log_level is empty, applying default.", "default", defaultLogLevel)
		c.LogLevel = defaultLogLevel
	} else {
		_, err := ParseLogLevel(c.LogLevel) // Use helper to validate
		if err != nil {
			logger.Warn("Config validation: Invalid log_level found, applying default.", "configured_value", c.LogLevel, "default", defaultLogLevel, "error", err)
			c.LogLevel = defaultLogLevel
		}
	}
	// Add checks for other fields if necessary

	if len(validationErrors) > 0 {
		return errors.Join(validationErrors...)
	}
	return nil
}

// FileConfig represents the structure of the JSON config file for unmarshalling.
// Uses pointers to distinguish between unset fields and zero-value fields.
type FileConfig struct {
	OllamaURL      *string   `json:"ollama_url"`
	Model          *string   `json:"model"`
	MaxTokens      *int      `json:"max_tokens"`
	Stop           *[]string `json:"stop"`
	Temperature    *float64  `json:"temperature"`
	LogLevel       *string   `json:"log_level"`
	UseAst         *bool     `json:"use_ast"`
	UseFim         *bool     `json:"use_fim"`
	MaxPreambleLen *int      `json:"max_preamble_len"`
	MaxSnippetLen  *int      `json:"max_snippet_len"`
}

// Diagnostic Structures (Internal representation)
type DiagnosticSeverity int

const (
	SeverityError   DiagnosticSeverity = 1
	SeverityWarning DiagnosticSeverity = 2
	SeverityInfo    DiagnosticSeverity = 3
	SeverityHint    DiagnosticSeverity = 4
)

type Position struct {
	Line      int // 0-based
	Character int // 0-based, byte offset within the line
}

type Range struct {
	Start Position
	End   Position
}

type Diagnostic struct {
	Range    Range
	Severity DiagnosticSeverity
	Code     string
	Source   string
	Message  string
}

// AstContextInfo holds structured information extracted from code analysis.
type AstContextInfo struct {
	FilePath           string
	Version            int
	CursorPos          token.Pos
	PackageName        string
	TargetPackage      *packages.Package
	TargetFileSet      *token.FileSet
	TargetAstFile      *ast.File
	EnclosingFunc      *types.Func
	EnclosingFuncNode  *ast.FuncDecl
	ReceiverType       string
	EnclosingBlock     *ast.BlockStmt
	Imports            []*ast.ImportSpec
	CommentsNearCursor []string
	IdentifierAtCursor *ast.Ident
	IdentifierType     types.Type
	IdentifierObject   types.Object
	IdentifierDefNode  ast.Node
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
	Diagnostics        []Diagnostic
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
}

// CachedAnalysisEntry represents the full structure stored in bbolt.
type CachedAnalysisEntry struct {
	SchemaVersion   int
	GoModHash       string
	InputFileHashes map[string]string
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
// Exported Errors
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
	ErrInvalidURI           = errors.New("invalid document URI")
)

// =============================================================================
// Interfaces for Components
// =============================================================================

type LLMClient interface {
	GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error)
}

type Analyzer interface {
	Analyze(ctx context.Context, filename string, version int, line, col int) (*AstContextInfo, error)
	Close() error
	InvalidateCache(dir string) error
	InvalidateMemoryCacheForURI(uri string, version int) error
	MemoryCacheEnabled() bool
	GetMemoryCacheMetrics() *ristretto.Metrics
}

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
		Stop:           []string{DefaultStop, "}", "//", "/*"},
		Temperature:    defaultTemperature,
		LogLevel:       defaultLogLevel,
		UseAst:         true,
		UseFim:         false,
		MaxPreambleLen: 2048,
		MaxSnippetLen:  2048,
	}
)

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadConfig loads configuration from standard locations, merges with defaults,
// and attempts to write a default config if none exists or is invalid.
func LoadConfig() (Config, error) {
	logger := slog.Default()
	cfg := DefaultConfig
	var loadedFromFile bool
	var loadErrors []error
	var configParseError error

	primaryPath, secondaryPath, pathErr := getConfigPaths()
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		logger.Warn("Could not determine config paths", "error", pathErr)
	}

	if primaryPath != "" {
		logger.Debug("Attempting to load config", "path", primaryPath)
		loaded, loadErr := loadAndMergeConfig(primaryPath, &cfg, logger)
		if loadErr != nil {
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", primaryPath, loadErr))
			logger.Warn("Failed to load or merge config", "path", primaryPath, "error", loadErr)
		} else if loaded {
			loadedFromFile = true
			logger.Info("Loaded config", "path", primaryPath)
		} else {
			logger.Debug("Config file not found or empty", "path", primaryPath)
		}
	}

	primaryNotFoundOrFailed := !loadedFromFile || configParseError != nil
	if primaryNotFoundOrFailed && secondaryPath != "" {
		logger.Debug("Attempting to load config from secondary path", "path", secondaryPath)
		loaded, loadErr := loadAndMergeConfig(secondaryPath, &cfg, logger)
		if loadErr != nil {
			if configParseError == nil && strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", secondaryPath, loadErr))
			logger.Warn("Failed to load or merge config", "path", secondaryPath, "error", loadErr)
		} else if loaded {
			if !loadedFromFile {
				loadedFromFile = true
				logger.Info("Loaded config", "path", secondaryPath)
			}
		} else {
			logger.Debug("Config file not found or empty", "path", secondaryPath)
		}
	}

	loadSucceeded := loadedFromFile && configParseError == nil
	if !loadSucceeded {
		if configParseError != nil {
			logger.Warn("Existing config file failed to parse. Attempting to write default.", "error", configParseError)
		} else {
			logger.Info("No valid config file found. Attempting to write default.")
		}
		writePath := primaryPath
		if writePath == "" {
			writePath = secondaryPath
		}

		if writePath != "" {
			logger.Info("Attempting to write default config", "path", writePath)
			if err := writeDefaultConfig(writePath, DefaultConfig); err != nil {
				logger.Warn("Failed to write default config", "path", writePath, "error", err)
				loadErrors = append(loadErrors, fmt.Errorf("writing default config failed: %w", err))
			}
		} else {
			logger.Warn("Cannot determine path to write default config.")
			loadErrors = append(loadErrors, errors.New("cannot determine default config path"))
		}
		cfg = DefaultConfig
		logger.Info("Using default configuration values.")
	}

	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	finalCfg := cfg
	if err := finalCfg.Validate(); err != nil {
		logger.Warn("Config after load/merge failed validation. Falling back to pure defaults.", "error", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		if valErr := DefaultConfig.Validate(); valErr != nil {
			logger.Error("FATAL: Default config is invalid", "error", valErr)
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
		slog.Warn("Could not determine user config directory", "error", cfgErr)
	}
	homeDir, homeErr := os.UserHomeDir()
	if homeErr == nil {
		secondary = filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
		if primary == "" && cfgErr != nil {
			primary = secondary
			slog.Debug("Using fallback primary config path", "path", primary)
			secondary = ""
		}
		if primary == secondary {
			secondary = ""
		}
	} else {
		slog.Warn("Could not determine user home directory", "error", homeErr)
	}
	if primary == "" && secondary == "" {
		err = fmt.Errorf("cannot determine config/home directories: config error: %v; home error: %v", cfgErr, homeErr)
	}
	return primary, secondary, err
}

// loadAndMergeConfig attempts to load config from a path and merge into cfg.
func loadAndMergeConfig(path string, cfg *Config, logger *slog.Logger) (loaded bool, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("reading config file %q failed: %w", path, err)
	}
	if len(data) == 0 {
		logger.Warn("Config file exists but is empty, ignoring.", "path", path)
		return true, nil
	}

	var fileCfg FileConfig
	if err := json.Unmarshal(data, &fileCfg); err != nil {
		return true, fmt.Errorf("parsing config file JSON %q failed: %w", path, err)
	}

	mergedFields := 0
	if fileCfg.OllamaURL != nil {
		cfg.OllamaURL = *fileCfg.OllamaURL
		mergedFields++
	}
	if fileCfg.Model != nil {
		cfg.Model = *fileCfg.Model
		mergedFields++
	}
	if fileCfg.MaxTokens != nil {
		cfg.MaxTokens = *fileCfg.MaxTokens
		mergedFields++
	}
	if fileCfg.Stop != nil {
		cfg.Stop = *fileCfg.Stop
		mergedFields++
	}
	if fileCfg.Temperature != nil {
		cfg.Temperature = *fileCfg.Temperature
		mergedFields++
	}
	if fileCfg.LogLevel != nil {
		cfg.LogLevel = *fileCfg.LogLevel
		mergedFields++
	}
	if fileCfg.UseAst != nil {
		cfg.UseAst = *fileCfg.UseAst
		mergedFields++
	}
	if fileCfg.UseFim != nil {
		cfg.UseFim = *fileCfg.UseFim
		mergedFields++
	}
	if fileCfg.MaxPreambleLen != nil {
		cfg.MaxPreambleLen = *fileCfg.MaxPreambleLen
		mergedFields++
	}
	if fileCfg.MaxSnippetLen != nil {
		cfg.MaxSnippetLen = *fileCfg.MaxSnippetLen
		mergedFields++
	}
	logger.Debug("Merged configuration from file", "path", path, "fields_merged", mergedFields)

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
		LogLevel       string   `json:"log_level"`
		UseAst         bool     `json:"use_ast"`
		UseFim         bool     `json:"use_fim"`
		MaxPreambleLen int      `json:"max_preamble_len"`
		MaxSnippetLen  int      `json:"max_snippet_len"`
	}
	expCfg := ExportableConfig{
		OllamaURL: defaultConfig.OllamaURL, Model: defaultConfig.Model, MaxTokens: defaultConfig.MaxTokens,
		Stop: defaultConfig.Stop, Temperature: defaultConfig.Temperature, LogLevel: defaultConfig.LogLevel,
		UseAst: defaultConfig.UseAst, UseFim: defaultConfig.UseFim,
		MaxPreambleLen: defaultConfig.MaxPreambleLen, MaxSnippetLen: defaultConfig.MaxSnippetLen,
	}
	jsonData, err := json.MarshalIndent(expCfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal default config to JSON: %w", err)
	}
	if err := os.WriteFile(path, jsonData, 0640); err != nil {
		return fmt.Errorf("failed to write default config file %s: %w", path, err)
	}
	slog.Info("Wrote default configuration", "path", path)
	return nil
}

// ParseLogLevel converts a string level ("debug", "info", "warn", "error") to slog.Level.
func ParseLogLevel(levelStr string) (slog.Level, error) {
	switch strings.ToLower(levelStr) {
	case "debug":
		return slog.LevelDebug, nil
	case "info":
		return slog.LevelInfo, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error", "err":
		return slog.LevelError, nil
	default:
		return slog.LevelInfo, fmt.Errorf("invalid log level string: %q (expected debug, info, warn, or error)", levelStr)
	}
}

// =============================================================================
// Default Component Implementations
// =============================================================================

// httpOllamaClient implements LLMClient using HTTP requests to Ollama.
type httpOllamaClient struct {
	httpClient *http.Client
}

func newHttpOllamaClient() *httpOllamaClient {
	return &httpOllamaClient{
		httpClient: &http.Client{
			Timeout: 90 * time.Second,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: 10 * time.Second,
				}).DialContext,
				TLSHandshakeTimeout: 10 * time.Second,
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
		if errors.Is(err, context.Canceled) {
			return nil, context.Canceled
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return nil, fmt.Errorf("%w: network timeout connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		if opErr, ok := err.(*net.OpError); ok && opErr.Op == "dial" {
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

// GoPackagesAnalyzer implements Analyzer using go/packages and caching.
type GoPackagesAnalyzer struct {
	db          *bbolt.DB
	memoryCache *ristretto.Cache
	mu          sync.Mutex
}

// NewGoPackagesAnalyzer initializes the analyzer and caches.
func NewGoPackagesAnalyzer() *GoPackagesAnalyzer {
	logger := slog.Default() // Use default logger
	dbPath := ""
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil {
			dbPath = filepath.Join(dbDir, "analysis_cache.db")
		} else {
			logger.Warn("Could not create bbolt cache directory", "path", dbDir, "error", err)
		}
	} else {
		logger.Warn("Could not determine user cache directory. Bbolt caching disabled.", "error", err)
	}

	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second}
		db, err = bbolt.Open(dbPath, 0600, opts)
		if err != nil {
			logger.Warn("Failed to open bbolt cache file. Bbolt caching will be disabled.", "path", dbPath, "error", err)
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
				logger.Warn("Failed to ensure bbolt bucket exists. Bbolt caching disabled.", "error", err)
				db.Close()
				db = nil
			} else {
				logger.Info("Using bbolt cache", "path", dbPath, "schema", cacheSchemaVersion)
			}
		}
	}

	memCache, cacheErr := ristretto.NewCache(&ristretto.Config{
		NumCounters: 1e7,
		MaxCost:     1 << 30,
		BufferItems: 64,
		Metrics:     true,
	})
	if cacheErr != nil {
		logger.Warn("Failed to create ristretto memory cache. In-memory caching disabled.", "error", cacheErr)
		memCache = nil
	} else {
		logger.Info("Initialized ristretto in-memory cache", "max_cost", "1GB")
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
		a.memoryCache.Close()
		a.memoryCache = nil
	}
	if len(closeErrors) > 0 {
		return errors.Join(closeErrors...)
	}
	return nil
}

// Analyze performs code analysis, orchestrating calls to helpers.
// Cycle 2 Fix: Correctly handle the 5 return values from loadPackageAndFile.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, absFilename string, version int, line, col int) (info *AstContextInfo, analysisErr error) {
	logger := slog.Default().With("absFile", absFilename, "version", version, "line", line, "col", col)
	info = &AstContextInfo{
		FilePath:         absFilename,
		Version:          version,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		Diagnostics:      make([]Diagnostic, 0),
		CallArgIndex:     -1,
	}
	defer func() {
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			logger.Error("Panic recovered during Analyze", "error", r, "stack", string(debug.Stack()))
			addAnalysisError(info, panicErr, logger)
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr)
			}
		}
		if len(info.AnalysisErrors) > 0 {
			finalErr := errors.Join(info.AnalysisErrors...)
			if analysisErr == nil {
				analysisErr = fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr)
			} else {
				analysisErr = fmt.Errorf("%w: %w", analysisErr, finalErr)
			}
		}
	}()

	logger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)
	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry
	var loadDuration, stepsDuration, preambleDuration time.Duration

	if a.db != nil {
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
			if err := gob.NewDecoder(bytes.NewReader(valBytes)).Decode(&decoded); err != nil {
				return fmt.Errorf("%w: %w", ErrCacheDecode, err)
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				logger.Warn("Cache data has old schema version. Ignoring.", "key", string(cacheKey))
				return nil
			}
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

		if cachedEntry != nil {
			validationStart := time.Now()
			logger.Debug("Potential bbolt cache hit. Validating file hashes...", "key", string(cacheKey))
			currentHashes, hashErr := calculateInputHashes(dir, nil)
			if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes) {
				logger.Debug("Bbolt cache VALID. Attempting to decode analysis data...", "key", string(cacheKey))
				decodeStart := time.Now()
				var analysisData CachedAnalysisData
				if decodeErr := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob)).Decode(&analysisData); decodeErr == nil {
					info.PackageName = analysisData.PackageName
					info.PromptPreamble = analysisData.PromptPreamble
					cacheHit = true
					loadDuration = time.Since(decodeStart)
					logger.Debug("Analysis data successfully decoded from bbolt cache.", "duration", loadDuration)
					logger.Debug("Re-running load/analysis steps for diagnostics despite cache hit.")
					cacheHit = false // Force re-analysis for diagnostics
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

	if !cacheHit {
		logger.Debug("Bbolt cache miss or invalid (or re-running for diagnostics). Performing full analysis...", "key", string(cacheKey))
		loadStart := time.Now()
		fset := token.NewFileSet()
		info.TargetFileSet = fset

		// Cycle 2 Fix: Capture all 5 return values from loadPackageAndFile
		var loadDiagnostics []Diagnostic
		var loadErrors []error
		targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := loadPackageAndFile(ctx, absFilename, fset, logger)
		info.Diagnostics = append(info.Diagnostics, loadDiagnostics...) // Append diagnostics from loading

		loadDuration = time.Since(loadStart)
		logger.Debug("packages.Load completed", "duration", loadDuration)
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr, logger)
			// Diagnostics already created in loadPackageAndFile
		}
		info.TargetPackage = targetPkg
		info.TargetAstFile = targetFileAST

		stepsStart := time.Now()
		if targetFile != nil {
			analyzeStepErr := performAnalysisSteps(targetFile, targetFileAST, targetPkg, fset, line, col, a, info, logger)
			if analyzeStepErr != nil {
				addAnalysisError(info, analyzeStepErr, logger)
			}
		} else {
			if len(loadErrors) == 0 {
				addAnalysisError(info, errors.New("cannot perform analysis steps: target token.File is nil"), logger)
			}
			gatherScopeContext(nil, targetPkg, fset, info, logger)
		}
		stepsDuration = time.Since(stepsStart)
		logger.Debug("Analysis steps completed", "duration", stepsDuration)

		if info.PromptPreamble == "" {
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
			info.PromptPreamble = constructPromptPreamble(a, info, qualifier, logger)
			preambleDuration = time.Since(preambleStart)
			logger.Debug("Preamble construction completed", "duration", preambleDuration)
		} else {
			logger.Debug("Skipping preamble construction (loaded from cache or already built).")
		}

		shouldSave := a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0
		if shouldSave {
			logger.Debug("Attempting to save analysis results (preamble) to bbolt cache.", "key", string(cacheKey))
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
							logger.Debug("Saved analysis results (preamble) to bbolt cache", "key", string(cacheKey), "duration", time.Since(saveStart))
						} else {
							logger.Warn("Failed to write to bbolt cache", "key", string(cacheKey), "error", saveErr)
							addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheWrite, saveErr), logger)
						}
					} else {
						addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr), logger)
					}
				} else {
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr), logger)
				}
			} else {
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger)
			}
		} else if a.db != nil {
			logger.Debug("Skipping bbolt cache save", "key", string(cacheKey), "load_errors", len(loadErrors), "preamble_empty", info.PromptPreamble == "")
		}
	}

	if info.PromptPreamble != "" && len(info.Diagnostics) == 0 && analysisErr == nil {
		logger.Info("Context analysis finished (cache hit for preamble, re-analyzed for diagnostics)", "decode_duration", loadDuration)
	} else {
		logger.Info("Context analysis finished (full analysis)", "load_duration", loadDuration, "steps_duration", stepsDuration, "preamble_duration", preambleDuration)
	}
	logger.Debug("Final Context Preamble generated", "length", len(info.PromptPreamble))
	logger.Debug("Final Diagnostics collected", "count", len(info.Diagnostics))

	return info, analysisErr
}

// InvalidateCache removes the bbolt cached entry for a given directory.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	logger := slog.Default().With("dir", dir)
	a.mu.Lock()
	db := a.db
	a.mu.Unlock()
	if db == nil {
		logger.Debug("Bbolt cache invalidation skipped: DB is nil.")
		return nil
	}
	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	logger.Info("Invalidating bbolt cache entry", "key", string(cacheKey))
	return deleteCacheEntryByKey(db, cacheKey, logger)
}

// InvalidateMemoryCacheForURI clears relevant entries from the ristretto cache.
func (a *GoPackagesAnalyzer) InvalidateMemoryCacheForURI(uri string, version int) error {
	logger := slog.Default().With("uri", uri, "version", version)
	a.mu.Lock()
	memCache := a.memoryCache
	a.mu.Unlock()

	if memCache == nil {
		logger.Debug("Memory cache invalidation skipped: Cache is nil.")
		return nil
	}
	logger.Warn("Clearing entire Ristretto memory cache due to document change.", "uri", uri)
	memCache.Clear()
	return nil
}

// MemoryCacheEnabled checks if the ristretto cache is active.
func (a *GoPackagesAnalyzer) MemoryCacheEnabled() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.memoryCache != nil
}

// GetMemoryCacheMetrics returns the ristretto cache metrics.
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
	template := config.PromptTemplate
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2
	logger := slog.Default() // Use default logger

	if len(contextPreamble) > maxPreambleLen {
		logger.Warn("Truncating context preamble", "original_length", len(contextPreamble), "max_length", maxPreambleLen)
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
			logger.Warn("Truncating FIM prefix", "original_length", len(prefix), "max_length", maxFIMPartLen)
			marker := "...(prefix truncated)"
			startByte := len(prefix) - maxFIMPartLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			prefix = marker + prefix[startByte:]
		}
		if len(suffix) > maxFIMPartLen {
			logger.Warn("Truncating FIM suffix", "original_length", len(suffix), "max_length", maxFIMPartLen)
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
			logger.Warn("Truncating code snippet (prefix)", "original_length", len(snippet), "max_length", maxSnippetLen)
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
	configMu  sync.RWMutex
}

// NewDeepCompleter creates a new DeepCompleter service with default config.
func NewDeepCompleter() (*DeepCompleter, error) {
	cfg, configErr := LoadConfig()
	logger := slog.Default()

	if configErr != nil && errors.Is(configErr, ErrConfig) {
		logger.Warn("Non-fatal error during initial config load. Using loaded/default values.", "error", configErr)
	} else if configErr != nil {
		logger.Error("Fatal error during initial config load", "error", configErr)
		return nil, configErr
	}

	if err := cfg.Validate(); err != nil {
		logger.Error("Loaded/default config is invalid even after LoadConfig", "error", err)
		return nil, fmt.Errorf("initial config validation failed: %w", err)
	}

	analyzer := NewGoPackagesAnalyzer()
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    cfg,
	}

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
	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    config,
	}, nil
}

// Close cleans up resources.
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
	slog.Info("DeepCompleter configuration updated",
		slog.String("ollama_url", newConfig.OllamaURL),
		slog.String("model", newConfig.Model),
		slog.Int("max_tokens", newConfig.MaxTokens),
		slog.Any("stop", newConfig.Stop),
		slog.Float64("temperature", newConfig.Temperature),
		slog.String("log_level", newConfig.LogLevel),
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
func (dc *DeepCompleter) InvalidateMemoryCacheForURI(uri string, version int) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	return dc.analyzer.InvalidateMemoryCacheForURI(uri, version)
}

// GetAnalyzer returns the analyzer instance.
func (dc *DeepCompleter) GetAnalyzer() Analyzer {
	return dc.analyzer
}

// GetCompletion provides basic completion for a direct code snippet.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	logger := slog.Default().With("operation", "GetCompletion")
	logger.Info("Handling basic completion request")
	currentConfig := dc.GetCurrentConfig()
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	logger.Debug("Generated basic prompt", "length", len(prompt))

	var buffer bytes.Buffer
	apiCallFunc := func() error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()
		logger.Debug("Calling Ollama GenerateStream for basic completion")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig)
		if apiErr != nil {
			return apiErr
		}
		streamCtx, cancelStream := context.WithTimeout(apiCtx, 50*time.Second)
		defer cancelStream()
		buffer.Reset()
		streamErr := streamCompletion(streamCtx, reader, &buffer)
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger)
	if err != nil {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			logger.Error("Ollama unavailable for basic completion after retries", "error", err)
			return "", fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error for basic completion after retries", "error", err)
			return "", err
		}
		logger.Error("Failed to get basic completion after retries", "error", err)
		return "", fmt.Errorf("failed to get basic completion after %d retries: %w", maxRetries, err)
	}

	logger.Info("Basic completion successful")
	return strings.TrimSpace(buffer.String()), nil
}

// GetCompletionStreamFromFile provides context-aware completion using analysis, streaming the result.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, absFilename string, version int, line, col int, w io.Writer) error {
	logger := slog.Default().With("operation", "GetCompletionStreamFromFile", "path", absFilename, "version", version, "line", line, "col", col)
	currentConfig := dc.GetCurrentConfig()
	var contextPreamble string = "// Basic file context only."
	var analysisInfo *AstContextInfo
	var analysisErr error

	if currentConfig.UseAst {
		logger.Info("Analyzing context (or checking cache)")
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second)
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, absFilename, version, line, col)
		cancelAnalysis()

		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			logger.Error("Fatal error during analysis/cache check", "error", analysisErr)
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		if analysisErr != nil {
			logger.Warn("Non-fatal error during analysis/cache check", "error", analysisErr)
		}
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

	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col)
	if snippetErr != nil {
		logger.Error("Failed to extract code snippet context", "error", snippetErr)
		return fmt.Errorf("failed to extract code snippet context: %w", snippetErr)
	}

	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	logger.Debug("Generated prompt", "length", len(prompt))

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
		}
		streamErr := streamCompletion(apiCtx, reader, w)
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger)
	if err != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			logger.Error("Ollama unavailable for stream after retries", "error", err)
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error for stream after retries", "error", err)
			return err
		}
		logger.Error("Failed to get completion stream after retries", "error", err)
		return fmt.Errorf("failed to get completion stream after %d retries: %w", maxRetries, err)
	}

	if analysisErr != nil {
		logger.Warn("Completion stream successful, but context analysis encountered non-fatal errors", "analysis_error", analysisErr)
	}
	logger.Info("Completion stream successful")
	return nil
}
