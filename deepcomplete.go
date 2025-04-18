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

	// "go/format" // No longer needed directly in this file
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

	// "sort" // No longer needed directly in this file
	"strings"
	"sync"
	"time"

	"go.etcd.io/bbolt"
	// "golang.org/x/tools/go/ast/astutil" // Moved to helpers
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
		slog.Warn("Could not determine config paths", "error", pathErr)
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
			slog.Info("Loaded config", "path", primaryPath)
		} else if loadedFromFile {
			slog.Warn("Attempted load but failed", "path", primaryPath, "error", loadErr)
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
			slog.Info("Loaded config", "path", secondaryPath)
		} else if loadedFromFile {
			slog.Warn("Attempted load but failed", "path", secondaryPath, "error", loadErr)
		}
	}

	loadSucceeded := loadedFromFile && configParseError == nil
	if !loadSucceeded {
		if configParseError != nil {
			slog.Warn("Existing config file failed to parse. Attempting to write default.", "error", configParseError)
		} else {
			slog.Info("No config file found. Attempting to write default.")
		}
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
		slog.Warn("Config after load/merge failed validation. Returning defaults.", "error", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		if valErr := DefaultConfig.Validate(); valErr != nil {
			slog.Error("Default config is invalid", "error", valErr)
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
func loadAndMergeConfig(path string, cfg *Config) (loaded bool, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return true, fmt.Errorf("reading config file %q failed: %w", path, err)
	}
	if len(data) == 0 {
		slog.Warn("Config file exists but is empty", "path", path)
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
			slog.Warn("Could not create bbolt cache directory", "path", dbDir, "error", err)
		}
	} else {
		slog.Warn("Could not determine user cache directory. Caching disabled.", "error", err)
	}

	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second}
		db, err = bbolt.Open(dbPath, 0600, opts)
		if err != nil {
			slog.Warn("Failed to open bbolt cache file. Caching will be disabled.", "path", dbPath, "error", err)
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
				slog.Warn("Failed to ensure bbolt bucket exists. Caching disabled.", "error", err)
				db.Close()
				db = nil
			} else {
				slog.Info("Using bbolt cache", "path", dbPath, "schema", cacheSchemaVersion)
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
		slog.Info("Closing bbolt cache database.")
		err := a.db.Close()
		a.db = nil
		return err
	}
	return nil
}

// Analyze performs code analysis, utilizing the cache if possible.
// Refactored in Cycle 8 to orchestrate calls to internal helpers.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, filename string, line, col int) (info *AstContextInfo, analysisErr error) {
	logger := slog.Default().With("file", filename, "line", line, "col", col)

	info = &AstContextInfo{FilePath: filename, VariablesInScope: make(map[string]types.Object), AnalysisErrors: make([]error, 0), CallArgIndex: -1}
	defer func() { // Recover from panics during analysis.
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			logger.Error("Panic recovered during AnalyzeCodeContext", "error", r, "stack", string(debug.Stack()))
			addAnalysisError(info, panicErr, logger) // Pass logger
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr)
			}
		}
	}()

	absFilename, err := filepath.Abs(filename)
	if err != nil {
		return info, fmt.Errorf("failed to get absolute path for '%s': %w", filename, err)
	}
	info.FilePath = absFilename
	logger = logger.With("absFile", absFilename)
	logger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)

	// --- Cache Check ---
	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	var loadDuration, stepsDuration, preambleDuration time.Duration
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry

	if a.db != nil {
		readStart := time.Now()
		dbErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				return fmt.Errorf("%w: cache bucket %s not found during View", ErrCacheRead, string(cacheBucketName))
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				return nil // Cache miss
			}
			var decoded CachedAnalysisEntry
			decoder := gob.NewDecoder(bytes.NewReader(valBytes))
			if err := decoder.Decode(&decoded); err != nil {
				logger.Warn("Failed to gob-decode cached entry header. Treating as miss.", "key", string(cacheKey), "error", err)
				deleteCacheEntryByKey(a.db, cacheKey, logger) // Pass logger
				return fmt.Errorf("%w: %w", ErrCacheDecode, err)
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				logger.Warn("Cache data has old schema version. Ignoring.", "key", string(cacheKey), "got_version", decoded.SchemaVersion, "want_version", cacheSchemaVersion)
				deleteCacheEntryByKey(a.db, cacheKey, logger) // Pass logger
				return nil                                    // Treat as miss
			}
			cachedEntry = &decoded
			return nil
		})
		if dbErr != nil {
			logger.Warn("Error reading or decoding from bbolt cache", "error", dbErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbErr), logger)
		}
		logger.Debug("Cache read attempt finished", "duration", time.Since(readStart))
	} else {
		logger.Debug("Cache disabled (db handle is nil).")
	}

	// Validate cache hit based on file hashes.
	if cachedEntry != nil {
		validationStart := time.Now()
		logger.Debug("Potential cache hit. Validating file hashes...", "key", string(cacheKey))
		currentHashes, hashErr := calculateInputHashes(dir, nil) // Pass nil pkg here.
		if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes) {
			logger.Debug("Cache VALID. Attempting to decode analysis data...", "key", string(cacheKey))
			decodeStart := time.Now()
			var analysisData CachedAnalysisData
			decoder := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob))
			if decodeErr := decoder.Decode(&analysisData); decodeErr == nil {
				info.PackageName = analysisData.PackageName
				info.PromptPreamble = analysisData.PromptPreamble
				cacheHit = true
				loadDuration = time.Since(decodeStart)
				logger.Debug("Analysis data successfully decoded from cache.", "duration", loadDuration)
				logger.Debug("Using cached preamble. Skipping packages.Load and analysis steps.", "preamble_length", len(info.PromptPreamble))
			} else {
				logger.Warn("Failed to gob-decode cached analysis data. Treating as miss.", "error", decodeErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheDecode, decodeErr), logger)
				deleteCacheEntryByKey(a.db, cacheKey, logger)
			}
		} else {
			logger.Debug("Cache INVALID. Treating as miss.", "key", string(cacheKey), "hash_error", hashErr)
			deleteCacheEntryByKey(a.db, cacheKey, logger)
			if hashErr != nil {
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger)
			}
		}
		logger.Debug("Cache validation/decode finished", "duration", time.Since(validationStart))
	}

	// --- Perform Full Analysis if Cache Miss ---
	if !cacheHit {
		if a.db == nil {
			logger.Debug("Cache disabled, loading via packages.Load...")
		} else {
			logger.Debug("Cache miss or invalid. Loading via packages.Load...", "key", string(cacheKey))
		}

		loadStart := time.Now()
		fset := token.NewFileSet()
		targetPkg, targetFileAST, targetFile, loadErrors := loadPackageInfo(ctx, absFilename, fset) // Uses slog internally now
		loadDuration = time.Since(loadStart)
		logger.Debug("packages.Load completed", "duration", loadDuration)
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr, logger) // Pass logger
		}

		if targetFile != nil && fset != nil {
			stepsStart := time.Now()
			// Cycle 8: Call refactored internal function (defined in analysis_helpers.go)
			analyzeStepErr := performAnalysisSteps(targetFile, targetFileAST, targetPkg, fset, line, col, info, logger) // Pass logger
			stepsDuration = time.Since(stepsStart)
			logger.Debug("performAnalysisSteps completed", "duration", stepsDuration)
			if analyzeStepErr != nil {
				addAnalysisError(info, analyzeStepErr, logger) // Pass logger
			}
		} else {
			if targetFile == nil {
				addAnalysisError(info, errors.New("cannot perform analysis steps: missing target file after load"), logger) // Pass logger
			}
			if fset == nil && targetFile != nil {
				addAnalysisError(info, errors.New("cannot perform analysis steps: missing FileSet after load"), logger) // Pass logger
			}
		}

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
			logger.Debug("Building preamble with limited/no type info.")
		}
		preambleStart := time.Now()
		// Cycle 8: Call refactored internal function (defined in analysis_helpers.go)
		info.PromptPreamble = buildPreamble(info, qualifier, logger) // Pass logger
		preambleDuration = time.Since(preambleStart)
		logger.Debug("buildPreamble completed", "duration", preambleDuration)

		// Save to cache if enabled, successful, and preamble generated.
		if a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0 {
			logger.Debug("Attempting to save analysis results to bbolt cache.", "key", string(cacheKey))
			saveStart := time.Now()
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg) // Uses slog internally now
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
							logger.Debug("Writing bytes to cache", "key", string(cacheKey), "bytes", len(encodedBytes))
							return b.Put(cacheKey, encodedBytes)
						})
						if saveErr == nil {
							logger.Debug("Saved analysis results to bbolt cache", "key", string(cacheKey), "duration", time.Since(saveStart))
						} else {
							logger.Warn("Failed to write to bbolt cache", "key", string(cacheKey), "error", saveErr)
							addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheWrite, saveErr), logger)
						}
					} else {
						logger.Warn("Failed to gob-encode cache entry", "key", string(cacheKey), "error", entryEncodeErr)
						addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr), logger)
					}
				} else {
					logger.Warn("Failed to gob-encode analysis data for caching", "error", encodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr), logger)
				}
			} else {
				logger.Warn("Failed to calculate input hashes for caching", "error", hashErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger)
			}
		} else if a.db != nil {
			logger.Debug("Skipping cache save", "key", string(cacheKey), "load_errors", len(loadErrors), "preamble_empty", info.PromptPreamble == "")
		}
	}

	logAnalysisErrors(info.AnalysisErrors, logger) // Pass logger
	analysisErr = errors.Join(info.AnalysisErrors...)
	if cacheHit {
		logger.Info("Context analysis finished", "decode_duration", loadDuration, "cache_hit", cacheHit)
	} else {
		logger.Info("Context analysis finished", "load_duration", loadDuration, "steps_duration", stepsDuration, "preamble_duration", preambleDuration, "cache_hit", cacheHit)
	}
	logger.Debug("Final Context Preamble generated", "length", len(info.PromptPreamble))

	// Return non-fatal analysis errors wrapped.
	if analysisErr != nil {
		return info, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisErr)
	}
	return info, nil
}

// InvalidateCache removes the cached entry for a given directory.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	logger := slog.Default().With("dir", dir)
	a.mu.Lock()
	db := a.db
	a.mu.Unlock()
	if db == nil {
		logger.Debug("Cache invalidation skipped: DB is nil.")
		return nil
	}
	goModHash := calculateGoModHash(dir) // Uses slog internally now
	cacheKey := []byte(dir + "::" + goModHash)
	return deleteCacheEntryByKey(db, cacheKey, logger) // Pass logger
}

// loadPackageInfo encapsulates the logic for loading package information.
// Assumes slog is initialized.
func loadPackageInfo(ctx context.Context, absFilename string, fset *token.FileSet) (*packages.Package, *ast.File, *token.File, []error) {
	var loadErrors []error
	dir := filepath.Dir(absFilename)
	logger := slog.Default().With("dir", dir, "file", absFilename)

	loadCfg := &packages.Config{
		Context: ctx,
		Dir:     dir,
		Fset:    fset,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedTypesSizes |
			packages.NeedSyntax | packages.NeedTypesInfo,
		Tests: false,
		Logf:  func(format string, args ...interface{}) { logger.Debug(fmt.Sprintf(format, args...)) }, // Pipe packages logs to slog
	}

	pkgs, err := packages.Load(loadCfg, fmt.Sprintf("file=%s", absFilename))
	if err != nil {
		loadErrors = append(loadErrors, fmt.Errorf("packages.Load failed: %w", err))
		logger.Error("packages.Load failed", "error", err)
		return nil, nil, nil, loadErrors
	}

	if len(pkgs) == 0 {
		loadErrors = append(loadErrors, errors.New("packages.Load returned no packages"))
		logger.Warn("packages.Load returned no packages.")
		return nil, nil, nil, loadErrors
	}

	var targetPkg *packages.Package
	var targetFileAST *ast.File
	var targetFile *token.File

	for _, p := range pkgs {
		for _, loadErr := range p.Errors {
			loadErrors = append(loadErrors, &loadErr)
			logger.Warn("Package loading error", "package", p.PkgPath, "error", loadErr)
		}
	}

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
				logger.Debug("Found target file in package", "package", p.PkgPath)
				goto foundTarget
			}
		}
	}

foundTarget:
	if targetPkg == nil {
		loadErrors = append(loadErrors, fmt.Errorf("target file %s not found in loaded packages", absFilename))
		logger.Warn("Target file not found in any loaded packages.")
		if len(pkgs) > 0 {
			targetPkg = pkgs[0]
			logger.Warn("Falling back to first loaded package, but target file AST/Token info will be missing.", "package", targetPkg.PkgPath)
		}
	}

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
// This function now orchestrates calls to helpers defined in analysis_helpers.go
func performAnalysisSteps(
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package,
	fset *token.FileSet,
	line, col int,
	info *AstContextInfo,
	logger *slog.Logger,
) error {
	if targetFile == nil || fset == nil {
		return errors.New("performAnalysisSteps requires non-nil targetFile and fset")
	}
	cursorPos, posErr := calculateCursorPos(targetFile, line, col) // Use helper from utils.go
	if posErr != nil {
		return fmt.Errorf("%w: %w", ErrPositionConversion, posErr)
	}
	info.CursorPos = cursorPos
	logger.Debug("Calculated cursor position", "pos", info.CursorPos, "pos_string", fset.PositionFor(info.CursorPos, true).String())

	if targetFileAST != nil {
		// Calls to helpers now assumed to be defined in analysis_helpers.go
		path := findEnclosingPath(targetFileAST, info.CursorPos, info, logger)
		findContextNodes(path, info.CursorPos, targetPkg, fset, info, logger)
		gatherScopeContext(path, targetPkg, fset, info, logger)
		findRelevantComments(targetFileAST, path, info.CursorPos, fset, info, logger)
	} else {
		addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil"), logger)
		gatherScopeContext(nil, targetPkg, fset, info, logger)
	}
	return nil
}

// --- Prompt Formatter ---
type templateFormatter struct{}

func newTemplateFormatter() *templateFormatter { return &templateFormatter{} }

// FormatPrompt combines context and snippet into the final LLM prompt, applying truncation.
// Assumes slog is initialized.
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2

	if len(contextPreamble) > maxPreambleLen {
		slog.Warn("Truncating context preamble", "original_length", len(contextPreamble), "max_length", maxPreambleLen)
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
			slog.Warn("Truncating FIM prefix", "original_length", len(prefix), "max_length", maxFIMPartLen)
			marker := "...(prefix truncated)"
			startByte := len(prefix) - maxFIMPartLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			prefix = marker + prefix[startByte:]
		}
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
		snippet := snippetCtx.Prefix
		if len(snippet) > maxSnippetLen {
			slog.Warn("Truncating code snippet", "original_length", len(snippet), "max_length", maxSnippetLen)
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
	analyzer  Analyzer // Keep unexported
	formatter PromptFormatter
	config    Config
	configMu  sync.RWMutex // Protects concurrent read/write access to config.
}

// NewDeepCompleter creates a new DeepCompleter service with default config.
// Assumes slog is initialized.
func NewDeepCompleter() (*DeepCompleter, error) {
	cfg, configErr := LoadConfig()
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		return nil, configErr
	}
	if configErr != nil {
		slog.Warn("Warning during initial config load", "error", configErr)
	}
	if err := cfg.Validate(); err != nil {
		slog.Warn("Initial config is invalid after load/merge. Using pure defaults.", "error", err)
		cfg = DefaultConfig
		if valErr := cfg.Validate(); valErr != nil {
			slog.Error("Default config validation failed", "error", valErr)
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
// Assumes slog is initialized.
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
// Assumes slog is initialized.
func (dc *DeepCompleter) Close() error {
	if dc.analyzer != nil {
		return dc.analyzer.Close()
	}
	return nil
}

// UpdateConfig atomically updates the completer's configuration after validation.
// Assumes slog is initialized.
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
	slog.Info("DeepCompleter configuration updated", "new_config", fmt.Sprintf("%+v", newConfig))
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

// InvalidateAnalyzerCache provides access to the analyzer's invalidation logic.
func (dc *DeepCompleter) InvalidateAnalyzerCache(dir string) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	return dc.analyzer.InvalidateCache(dir)
}

// GetCompletion provides basic completion for a direct code snippet (no AST analysis).
// Assumes slog is initialized.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	logger := slog.Default().With("operation", "GetCompletion")
	logger.Info("Handling basic completion request")

	dc.configMu.RLock()
	currentConfig := dc.config
	dc.configMu.RUnlock()

	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	logger.Debug("Generated basic prompt", "length", len(prompt))

	reader, err := dc.client.GenerateStream(ctx, prompt, currentConfig)
	if err != nil {
		logger.Error("LLM client GenerateStream failed", "error", err)
		return "", err
	}

	var buffer bytes.Buffer
	streamCtx, cancelStream := context.WithTimeout(ctx, 50*time.Second)
	defer cancelStream()

	if streamErr := streamCompletion(streamCtx, reader, &buffer); streamErr != nil { // streamCompletion uses slog
		logger.Error("Error processing LLM stream", "error", streamErr)
		if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
			return "", fmt.Errorf("%w: streaming context error: %w", ErrOllamaUnavailable, streamErr)
		}
		return "", fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
	}
	return strings.TrimSpace(buffer.String()), nil
}

// GetCompletionStreamFromFile provides context-aware completion using analysis, streaming the result.
// Assumes slog is initialized.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, filename string, row, col int, w io.Writer) error {
	logger := slog.Default().With("operation", "GetCompletionStreamFromFile", "file", filename, "line", row, "col", col)

	dc.configMu.RLock()
	currentConfig := dc.config
	dc.configMu.RUnlock()
	var contextPreamble string = "// Basic file context only."
	var analysisInfo *AstContextInfo
	var analysisErr error

	if currentConfig.UseAst {
		logger.Info("Analyzing context (or checking cache)")
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second)
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, filename, row, col) // Analyze uses slog
		cancelAnalysis()

		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) && !errors.Is(analysisErr, ErrCache) {
			logger.Error("Fatal error during analysis/cache check", "error", analysisErr)
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		if analysisErr != nil {
			logger.Warn("Non-fatal error during analysis/cache check", "error", analysisErr)
		}
		if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else if analysisErr != nil {
			contextPreamble += fmt.Sprintf("\n// Warning: Context analysis completed with errors and no preamble: %v\n", analysisErr)
		} else {
			contextPreamble += "\n// Warning: Context analysis returned no specific context preamble.\n"
		}
	} else {
		logger.Info("AST analysis disabled by config.")
	}

	snippetCtx, err := extractSnippetContext(filename, row, col) // extractSnippetContext uses slog
	if err != nil {
		logger.Error("Failed to extract code snippet context", "error", err)
		return fmt.Errorf("failed to extract code snippet context: %w", err)
	}

	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig) // FormatPrompt uses slog
	logger.Debug("Generated prompt", "length", len(prompt))

	apiCallFunc := func() error {
		select {
		case <-ctx.Done():
			logger.Warn("Context cancelled before API call attempt", "error", ctx.Err())
			return ctx.Err()
		default:
		}

		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()
		logger.Debug("Calling Ollama GenerateStream")
		apiStartTime := time.Now()
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig)
		apiDuration := time.Since(apiStartTime)
		logger.Debug("Ollama GenerateStream returned", "duration", apiDuration, "error", apiErr)

		if apiErr != nil {
			var oe *OllamaError
			isRetryable := errors.As(apiErr, &oe) && (oe.Status == http.StatusServiceUnavailable || oe.Status == http.StatusTooManyRequests)
			isRetryable = isRetryable || errors.Is(apiErr, context.DeadlineExceeded) || errors.Is(apiErr, ErrOllamaUnavailable)

			select {
			case <-ctx.Done():
				logger.Warn("Parent context cancelled after API call failed", "api_error", apiErr, "context_error", ctx.Err())
				return ctx.Err()
			default:
				if isRetryable {
					logger.Warn("Retryable API error", "error", apiErr)
					return apiErr
				}
				logger.Error("Non-retryable API error", "error", apiErr)
				return apiErr
			}
		}
		streamErr := streamCompletion(apiCtx, reader, w) // streamCompletion uses slog
		if streamErr != nil {
			select {
			case <-ctx.Done():
				logger.Warn("Parent context cancelled after stream error", "stream_error", streamErr, "context_error", ctx.Err())
				return ctx.Err()
			default:
				if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
					select {
					case <-ctx.Done():
						logger.Warn("Parent context cancelled after stream context error", "stream_error", streamErr, "context_error", ctx.Err())
						return ctx.Err()
					default:
						logger.Warn("Stream context error occurred", "error", streamErr)
						return streamErr
					}
				}
				logger.Error("Stream processing error", "error", streamErr)
				return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
			}
		}
		logger.Debug("Completion stream finished successfully for this attempt.")
		return nil
	}

	err = retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // retry uses slog
	if err != nil {
		select {
		case <-ctx.Done():
			logger.Warn("Context cancelled after retry attempts failed", "final_error", err, "context_error", ctx.Err())
			return ctx.Err()
		default:
		}
		var oe *OllamaError
		if errors.As(err, &oe) || errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			logger.Error("Ollama unavailable after retries", "error", err)
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error after retries", "error", err)
			return err
		}
		logger.Error("Failed to get completion stream after retries", "error", err, "retries", maxRetries)
		return fmt.Errorf("failed to get completion stream after %d retries: %w", maxRetries, err)
	}

	if analysisErr != nil {
		logger.Warn("Completion succeeded, but context analysis/cache check encountered non-fatal errors", "analysis_error", analysisErr)
	}
	logger.Info("Completion stream successful")
	return nil
}
