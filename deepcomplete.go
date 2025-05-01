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
	"go/token"
	"go/types"
	"io"
	stdslog "log/slog" // Use alias to avoid conflict if package name is slog
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/dgraph-io/ristretto"
	"go.etcd.io/bbolt"
)

// =============================================================================
// Constants & Global Variables
// =============================================================================

// Constants like default URLs, templates, etc., are now in types.go

var (
	cacheBucketName = []byte("AnalysisCache") // Name of the bbolt bucket for caching.
)

// Core type definitions (Config, AstContextInfo, Diagnostic, etc.) moved to types.go in Cycle 1.
// Exported error variables (Err*) moved to errors.go in Cycle 1.

// =============================================================================
// Interfaces for Components
// =============================================================================

// LLMClient defines the interface for interacting with the language model backend.
type LLMClient interface {
	// GenerateStream sends a prompt to the LLM and returns a stream of generated text.
	GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error)
}

// Analyzer defines the interface for code analysis.
type Analyzer interface {
	// Analyze performs static analysis on the given file at the specified position.
	Analyze(ctx context.Context, filename string, version int, line, col int) (*AstContextInfo, error)
	// Close cleans up any resources used by the analyzer (e.g., cache connections).
	Close() error
	// InvalidateCache removes cached data related to a specific directory (e.g., when go.mod changes).
	InvalidateCache(dir string) error
	// InvalidateMemoryCacheForURI removes memory-cached data for a specific file URI (e.g., on file change).
	InvalidateMemoryCacheForURI(uri string, version int) error
	// MemoryCacheEnabled returns true if the in-memory cache is configured and active.
	MemoryCacheEnabled() bool
	// GetMemoryCacheMetrics returns performance metrics for the in-memory cache.
	GetMemoryCacheMetrics() *ristretto.Metrics
}

// PromptFormatter defines the interface for constructing the final prompt sent to the LLM.
type PromptFormatter interface {
	// FormatPrompt combines the analysis preamble and code snippet context into the final prompt string.
	FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string
}

// =============================================================================
// Configuration Loading (Moved most logic to types.go)
// =============================================================================

// LoadConfig loads configuration from standard locations, merges with defaults,
// validates, and attempts to write a default config if needed.
// Returns the final Config and a non-fatal ErrConfig if warnings occurred.
// Cycle 1: Reduced scope, relies on helpers in types.go.
// Cycle 2: Simplify logging and error handling.
func LoadConfig(logger *stdslog.Logger) (Config, error) {
	if logger == nil {
		logger = stdslog.Default() // Use default if nil
	}
	cfg := getDefaultConfig() // Start with defaults (from types.go)
	var loadedFromFile bool
	var loadErrors []error
	var configParseError error

	primaryPath, secondaryPath, pathErr := getConfigPaths(logger) // Uses helper below
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		logger.Warn("Could not determine config paths, using defaults", "error", pathErr)
	}

	// Try primary path
	if primaryPath != "" {
		logger.Debug("Attempting to load config", "path", primaryPath)
		loaded, loadErr := loadAndMergeConfig(primaryPath, &cfg, logger) // Uses helper below
		if loadErr != nil {
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", primaryPath, loadErr))
			logger.Warn("Failed to load or merge config", "path", primaryPath, "error", loadErr)
		} else if loaded {
			loadedFromFile = true
			logger.Info("Loaded config", "path", primaryPath)
		}
	}

	// Try secondary path if primary failed or wasn't found
	primaryNotFoundOrFailed := !loadedFromFile || configParseError != nil
	if primaryNotFoundOrFailed && secondaryPath != "" && secondaryPath != primaryPath {
		logger.Debug("Attempting to load config from secondary path", "path", secondaryPath)
		loaded, loadErr := loadAndMergeConfig(secondaryPath, &cfg, logger)
		if loadErr != nil {
			if configParseError == nil && strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", secondaryPath, loadErr))
			logger.Warn("Failed to load or merge config", "path", secondaryPath, "error", loadErr)
		} else if loaded && !loadedFromFile {
			loadedFromFile = true
			logger.Info("Loaded config", "path", secondaryPath)
		}
	}

	// Write default config if no valid file was loaded
	loadSucceeded := loadedFromFile && configParseError == nil
	if !loadSucceeded {
		writePath := primaryPath // Prefer primary path for writing
		if writePath == "" {
			writePath = secondaryPath
		}

		if writePath != "" {
			if configParseError != nil {
				logger.Warn("Existing config file failed to parse. Attempting to write default.", "path", writePath, "error", configParseError)
			} else {
				logger.Info("No valid config file found. Attempting to write default.", "path", writePath)
			}
			if err := writeDefaultConfig(writePath, getDefaultConfig(), logger); err != nil { // Uses helper below
				logger.Warn("Failed to write default config", "path", writePath, "error", err)
				loadErrors = append(loadErrors, fmt.Errorf("writing default config failed: %w", err))
			}
		} else {
			logger.Warn("Cannot determine path to write default config.")
			loadErrors = append(loadErrors, errors.New("cannot determine default config path"))
		}
		cfg = getDefaultConfig() // Reset to defaults if write failed or no path found
		logger.Info("Using default configuration values.")
	}

	// Final validation of the resulting config (loaded or default)
	finalCfg := cfg
	// Validate method is defined on Config struct in types.go
	if err := finalCfg.Validate(logger); err != nil {
		logger.Error("Final configuration is invalid, falling back to pure defaults.", "error", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		pureDefault := getDefaultConfig()
		// Validate the pure DefaultConfig as a last resort
		if valErr := pureDefault.Validate(logger); valErr != nil {
			logger.Error("FATAL: Default config definition is invalid", "error", valErr)
			// Return the invalid default config but with a fatal error
			return pureDefault, fmt.Errorf("default config definition is invalid: %w", valErr)
		}
		finalCfg = pureDefault // Use pure defaults if merged/loaded config is invalid
	}

	// Return final config and wrap any accumulated non-fatal errors
	if len(loadErrors) > 0 {
		return finalCfg, fmt.Errorf("%w: %w", ErrConfig, errors.Join(loadErrors...))
	}
	return finalCfg, nil
}

// getConfigPaths determines the primary (XDG_CONFIG_HOME) and secondary (~/.config) config paths.
func getConfigPaths(logger *stdslog.Logger) (primary string, secondary string, err error) {
	var cfgErr, homeErr error

	// Primary Path: XDG_CONFIG_HOME
	userConfigDir, cfgErr := os.UserConfigDir()
	if cfgErr == nil {
		primary = filepath.Join(userConfigDir, configDirName, defaultConfigFileName)
	} else {
		logger.Warn("Could not determine user config directory (XDG)", "error", cfgErr)
	}

	// Secondary Path: ~/.config
	homeDir, homeErr := os.UserHomeDir()
	if homeErr == nil {
		secondaryCandidate := filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
		// If primary path failed, use secondary as primary
		if primary == "" && cfgErr != nil {
			primary = secondaryCandidate
			logger.Debug("Using fallback primary config path", "path", primary)
		} else if primary != secondaryCandidate {
			// Only set secondary if it's different from primary
			secondary = secondaryCandidate
		}
	} else {
		logger.Warn("Could not determine user home directory", "error", homeErr)
	}

	// If neither path could be determined, return an error
	if primary == "" {
		err = fmt.Errorf("cannot determine config/home directories: config error: %v; home error: %v", cfgErr, homeErr)
	}
	return primary, secondary, err
}

// loadAndMergeConfig attempts to load config from a specific path and merge its
// fields into the provided cfg object. Uses FileConfig struct from types.go.
func loadAndMergeConfig(path string, cfg *Config, logger *stdslog.Logger) (loaded bool, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil // File not found is not an error here
		}
		return false, fmt.Errorf("reading config file %q failed: %w", path, err)
	}
	loaded = true // File exists

	if len(data) == 0 {
		logger.Warn("Config file exists but is empty, ignoring.", "path", path)
		return loaded, nil // Empty file is not a parsing error
	}

	var fileCfg FileConfig // Use FileConfig from types.go
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&fileCfg); err != nil {
		return loaded, fmt.Errorf("parsing config file JSON %q failed: %w", path, err)
	}

	// Merge fields from fileCfg into cfg if they are non-nil
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
	return loaded, nil
}

// writeDefaultConfig creates the config directory if needed and writes the
// default configuration values as a JSON file.
func writeDefaultConfig(path string, defaultConfig Config, logger *stdslog.Logger) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0750); err != nil {
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}

	// Create an exportable struct containing only fields meant for the config file
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
		OllamaURL:      defaultConfig.OllamaURL,
		Model:          defaultConfig.Model,
		MaxTokens:      defaultConfig.MaxTokens,
		Stop:           defaultConfig.Stop,
		Temperature:    defaultConfig.Temperature,
		LogLevel:       defaultConfig.LogLevel,
		UseAst:         defaultConfig.UseAst,
		UseFim:         defaultConfig.UseFim,
		MaxPreambleLen: defaultConfig.MaxPreambleLen,
		MaxSnippetLen:  defaultConfig.MaxSnippetLen,
	}

	jsonData, err := json.MarshalIndent(expCfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal default config to JSON: %w", err)
	}

	if err := os.WriteFile(path, jsonData, 0640); err != nil { // Restricted permissions
		return fmt.Errorf("failed to write default config file %s: %w", path, err)
	}
	logger.Info("Wrote default configuration", "path", path)
	return nil
}

// ParseLogLevel converts a log level string to its corresponding slog.Level constant.
// Cycle 1: Moved from deepcomplete.go originally, now part of types.go implicitly via Config validation.
// Can be kept here as a public helper if needed elsewhere.
func ParseLogLevel(levelStr string) (stdslog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(levelStr)) {
	case "debug":
		return stdslog.LevelDebug, nil
	case "info":
		return stdslog.LevelInfo, nil
	case "warn", "warning":
		return stdslog.LevelWarn, nil
	case "error", "err":
		return stdslog.LevelError, nil
	default:
		return stdslog.LevelInfo, fmt.Errorf("invalid log level string: %q (expected debug, info, warn, or error)", levelStr)
	}
}

// =============================================================================
// Default Component Implementations
// =============================================================================

// --- Default LLM Client ---

// httpOllamaClient implements the LLMClient interface using HTTP requests to an Ollama server.
type httpOllamaClient struct {
	httpClient *http.Client
}

// newHttpOllamaClient creates a new Ollama client with reasonable timeouts.
func newHttpOllamaClient() *httpOllamaClient {
	return &httpOllamaClient{
		httpClient: &http.Client{
			Timeout: 90 * time.Second, // Overall request timeout
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: 10 * time.Second, // Connection timeout
				}).DialContext,
				TLSHandshakeTimeout: 10 * time.Second,
				MaxIdleConns:        10,
				IdleConnTimeout:     30 * time.Second,
			},
		},
	}
}

// GenerateStream sends a request to Ollama's /api/generate endpoint and returns the streaming response body.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	logger := stdslog.Default() // Use default logger
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
			"num_ctx":     4096, // Consider making this configurable if needed
			"top_p":       0.9,  // Common default, consider making configurable
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
	req.Header.Set("Accept", "application/x-ndjson") // Ollama uses newline-delimited JSON for streaming

	logger.Debug("Sending request to Ollama", "url", endpointURL, "model", config.Model)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Handle context cancellation/deadline explicitly
		if errors.Is(err, context.Canceled) {
			logger.Warn("Ollama request cancelled", "url", endpointURL)
			return nil, context.Canceled
		}
		if errors.Is(err, context.DeadlineExceeded) {
			logger.Error("Ollama request timed out", "url", endpointURL, "timeout", c.httpClient.Timeout)
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		// Handle network errors
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			logger.Error("Network timeout connecting to Ollama", "host", u.Host)
			return nil, fmt.Errorf("%w: network timeout connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		if opErr, ok := err.(*net.OpError); ok && opErr.Op == "dial" {
			logger.Error("Connection refused or network error connecting to Ollama", "host", u.Host)
			return nil, fmt.Errorf("%w: connection refused or network error connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		// General HTTP request failure
		logger.Error("HTTP request to Ollama failed", "url", endpointURL, "error", err)
		return nil, fmt.Errorf("%w: http request failed: %w", ErrOllamaUnavailable, err)
	}

	// Check for non-200 status codes
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(failed to read error response body)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			// Try to parse Ollama's specific error structure
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			// Use OllamaResponse from types.go
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error // Use the specific error message if available
			}
		}
		// Use OllamaError from types.go
		err = &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		logger.Error("Ollama API returned non-OK status", "status", resp.Status, "response_body", bodyString)
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
	}

	// Return the response body for streaming
	return resp.Body, nil
}

// --- Default Analyzer ---

// GoPackagesAnalyzer implements the Analyzer interface using go/packages and bbolt/ristretto caching.
type GoPackagesAnalyzer struct {
	db          *bbolt.DB        // Persistent disk cache (bbolt)
	memoryCache *ristretto.Cache // In-memory cache (ristretto)
	mu          sync.Mutex       // Protects access to db/memoryCache handles during Close/Invalidate
}

// NewGoPackagesAnalyzer initializes the analyzer, including setting up bbolt and ristretto caches.
func NewGoPackagesAnalyzer(logger *stdslog.Logger) *GoPackagesAnalyzer {
	if logger == nil {
		logger = stdslog.Default() // Use default if nil
	}
	dbPath := ""

	// Determine cache directory path
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		// Include schema version in the path to automatically invalidate old caches
		// Constant cacheSchemaVersion is defined in types.go
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil {
			dbPath = filepath.Join(dbDir, "analysis_cache.db")
		} else {
			logger.Warn("Could not create bbolt cache directory, disk caching disabled.", "path", dbDir, "error", err)
		}
	} else {
		logger.Warn("Could not determine user cache directory, disk caching disabled.", "error", err)
	}

	// Initialize bbolt database if path is valid
	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second}
		db, err = bbolt.Open(dbPath, 0600, opts)
		if err != nil {
			logger.Warn("Failed to open bbolt cache file, disk caching disabled.", "path", dbPath, "error", err)
			db = nil
		} else {
			// Ensure the cache bucket exists
			err = db.Update(func(tx *bbolt.Tx) error {
				_, err := tx.CreateBucketIfNotExists(cacheBucketName)
				if err != nil {
					return fmt.Errorf("failed to create cache bucket %s: %w", string(cacheBucketName), err)
				}
				return nil
			})
			if err != nil {
				logger.Warn("Failed to ensure bbolt bucket exists, disk caching disabled.", "error", err)
				db.Close()
				db = nil
			} else {
				logger.Info("Using bbolt disk cache", "path", dbPath, "schema_version", cacheSchemaVersion)
			}
		}
	}

	// Initialize Ristretto memory cache
	memCache, cacheErr := ristretto.NewCache(&ristretto.Config{
		NumCounters: 1e7,     // 10M keys recommended by Ristretto docs
		MaxCost:     1 << 30, // 1GB max cache size
		BufferItems: 64,
		Metrics:     true,
	})
	if cacheErr != nil {
		logger.Warn("Failed to create ristretto memory cache, in-memory caching disabled.", "error", cacheErr)
		memCache = nil
	} else {
		logger.Info("Initialized ristretto in-memory cache", "max_cost", "1GB")
	}

	return &GoPackagesAnalyzer{db: db, memoryCache: memCache}
}

// Close cleans up resources used by the analyzer, primarily closing cache connections.
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	var closeErrors []error
	logger := stdslog.Default()

	if a.db != nil {
		logger.Info("Closing bbolt cache database.")
		if err := a.db.Close(); err != nil {
			logger.Error("Error closing bbolt database", "error", err)
			closeErrors = append(closeErrors, fmt.Errorf("bbolt close failed: %w", err))
		}
		a.db = nil
	}
	if a.memoryCache != nil {
		logger.Info("Closing ristretto memory cache.")
		a.memoryCache.Close()
		a.memoryCache = nil
	}

	if len(closeErrors) > 0 {
		return errors.Join(closeErrors...)
	}
	return nil
}

// Analyze performs code analysis for a given file and position, utilizing caching.
// Returns the populated AstContextInfo and any fatal error encountered during loading/analysis.
// Non-fatal errors are collected within AstContextInfo.AnalysisErrors.
// Cycle 1: Uses AstContextInfo, CachedAnalysisEntry, etc., from types.go
// Cycle 3: Pass ctx to performAnalysisSteps and gatherScopeContext
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, absFilename string, version int, line, col int) (info *AstContextInfo, analysisErr error) {
	logger := stdslog.Default().With("absFile", absFilename, "version", version, "line", line, "col", col)
	// Use AstContextInfo from types.go
	info = &AstContextInfo{
		FilePath:         absFilename,
		Version:          version,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		Diagnostics:      make([]Diagnostic, 0), // Use Diagnostic from types.go
		CallArgIndex:     -1,
	}

	// Panic recovery defer function
	defer func() {
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			logger.Error("Panic recovered during Analyze", "error", r, "stack", string(debug.Stack()))
			addAnalysisError(info, panicErr, logger) // Use helper from helpers_diagnostics.go
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr)
			}
		}
		// Wrap non-fatal errors if no fatal error occurred
		if len(info.AnalysisErrors) > 0 && analysisErr == nil {
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr) // Use ErrAnalysisFailed from errors.go
		} else if len(info.AnalysisErrors) > 0 && analysisErr != nil {
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", analysisErr, finalErr)
		}
	}()

	logger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)
	goModHash := calculateGoModHash(dir) // Helper from utils.go
	cacheKey := []byte(dir + "::" + goModHash)
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry // Use CachedAnalysisEntry from types.go
	var loadDuration, stepsDuration, preambleDuration time.Duration

	// --- Bbolt Cache Check ---
	if a.db != nil {
		readStart := time.Now()
		dbViewErr := a.db.View(func(tx *bbolt.Tx) error {
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

			logger.Debug("Bbolt cache hit (raw bytes found). Decoding entry...", "key", string(cacheKey))
			var decoded CachedAnalysisEntry // Use CachedAnalysisEntry from types.go
			if err := gob.NewDecoder(bytes.NewReader(valBytes)).Decode(&decoded); err != nil {
				return fmt.Errorf("%w: %w", ErrCacheDecode, err) // Use ErrCacheDecode from errors.go
			}
			// Use cacheSchemaVersion from types.go
			if decoded.SchemaVersion != cacheSchemaVersion {
				logger.Warn("Bbolt cache data has old schema version. Ignoring.", "key", string(cacheKey), "cached_version", decoded.SchemaVersion, "expected_version", cacheSchemaVersion)
				return nil // Treat as miss
			}
			cachedEntry = &decoded
			return nil
		})
		// Handle errors from the View operation
		if dbViewErr != nil {
			logger.Warn("Error reading or decoding from bbolt cache. Cache check failed.", "error", dbViewErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbViewErr), logger) // Use ErrCacheRead from errors.go
			if errors.Is(dbViewErr, ErrCacheDecode) {                                     // Use ErrCacheDecode from errors.go
				go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "decode_failure")) // Helper from utils.go
			}
			cachedEntry = nil
		}
		logger.Debug("Bbolt cache read attempt finished", "duration", time.Since(readStart))

		// --- Bbolt Cache Validation ---
		if cachedEntry != nil {
			validationStart := time.Now()
			logger.Debug("Potential bbolt cache hit. Validating file hashes...", "key", string(cacheKey))
			currentHashes, hashErr := calculateInputHashes(dir, nil)                                                                   // Helper from utils.go
			if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes) { // Helper from utils.go
				logger.Debug("Bbolt cache VALID. Attempting to decode analysis data...", "key", string(cacheKey))
				decodeStart := time.Now()
				var analysisData CachedAnalysisData // Use CachedAnalysisData from types.go
				if decodeErr := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob)).Decode(&analysisData); decodeErr == nil {
					info.PackageName = analysisData.PackageName
					info.PromptPreamble = analysisData.PromptPreamble
					cacheHit = true
					loadDuration = time.Since(decodeStart)
					logger.Debug("Analysis data successfully decoded from bbolt cache.", "duration", loadDuration)
					// Force re-analysis for diagnostics/hover despite cache hit
					logger.Debug("Re-running load/analysis steps for diagnostics/hover despite cache hit.")
					cacheHit = false
				} else {
					logger.Warn("Failed to gob-decode cached analysis data. Treating as miss.", "error", decodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheDecode, decodeErr), logger) // Use ErrCacheDecode
					go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "analysis_decode_failure"))
					cacheHit = false
				}
			} else {
				logger.Debug("Bbolt cache INVALID (hash mismatch or error). Treating as miss.", "key", string(cacheKey), "hash_error", hashErr)
				go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "hash_mismatch"))
				if hashErr != nil {
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger) // Use ErrCacheHash from errors.go
				}
				cacheHit = false
			}
			logger.Debug("Bbolt cache validation/decode finished", "duration", time.Since(validationStart))
		}
	} else {
		logger.Debug("Bbolt cache disabled (db handle is nil).")
	}

	// --- Perform Full Analysis if Cache Miss or Invalid ---
	if !cacheHit {
		logger.Debug("Cache miss or invalid. Performing full analysis...", "key", string(cacheKey))
		loadStart := time.Now()
		fset := token.NewFileSet()
		info.TargetFileSet = fset

		// Use Diagnostic from types.go
		var loadDiagnostics []Diagnostic
		var loadErrors []error
		targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := loadPackageAndFile(ctx, absFilename, fset, logger) // From helpers_loader.go
		info.Diagnostics = append(info.Diagnostics, loadDiagnostics...)

		loadDuration = time.Since(loadStart)
		logger.Debug("packages.Load completed", "duration", loadDuration)
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr, logger)
		}
		info.TargetPackage = targetPkg
		info.TargetAstFile = targetFileAST

		stepsStart := time.Now()
		if targetFile != nil {
			analyzeStepErr := performAnalysisSteps(ctx, targetFile, targetFileAST, targetPkg, fset, line, col, a, info, logger) // From helpers_analysis_steps.go
			if analyzeStepErr != nil {
				addAnalysisError(info, analyzeStepErr, logger)
			}
		} else {
			if len(loadErrors) == 0 {
				addAnalysisError(info, errors.New("cannot perform analysis steps: target token.File is nil"), logger)
			}
			gatherScopeContext(ctx, nil, targetPkg, fset, info, logger) // From helpers_analysis_steps.go
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
				logger.Debug("Building preamble with limited or no type info.")
			}
			// Use AstContextInfo from types.go
			info.PromptPreamble = constructPromptPreamble(a, info, qualifier, logger) // From helpers_preamble.go
			preambleDuration = time.Since(preambleStart)
			logger.Debug("Preamble construction completed", "duration", preambleDuration)
		} else {
			logger.Debug("Skipping preamble construction.")
		}

		// --- Bbolt Cache Write ---
		shouldSave := a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0
		if shouldSave {
			logger.Debug("Attempting to save analysis results to bbolt cache.", "key", string(cacheKey))
			saveStart := time.Now()
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg) // Helper from utils.go
			if hashErr == nil {
				// Use CachedAnalysisData from types.go
				analysisDataToCache := CachedAnalysisData{
					PackageName:    info.PackageName,
					PromptPreamble: info.PromptPreamble,
				}
				var gobBuf bytes.Buffer
				if encodeErr := gob.NewEncoder(&gobBuf).Encode(&analysisDataToCache); encodeErr == nil {
					analysisGob := gobBuf.Bytes()
					// Use CachedAnalysisEntry from types.go
					entryToSave := CachedAnalysisEntry{
						SchemaVersion:   cacheSchemaVersion, // From types.go
						GoModHash:       goModHash,
						InputFileHashes: inputHashes,
						AnalysisGob:     analysisGob,
					}
					var entryBuf bytes.Buffer
					if entryEncodeErr := gob.NewEncoder(&entryBuf).Encode(&entryToSave); entryEncodeErr == nil {
						encodedBytes := entryBuf.Bytes()
						saveErr := a.db.Update(func(tx *bbolt.Tx) error {
							b := tx.Bucket(cacheBucketName)
							if b == nil {
								return fmt.Errorf("%w: cache bucket %s disappeared", ErrCacheWrite, string(cacheBucketName)) // Use ErrCacheWrite from errors.go
							}
							logger.Debug("Writing bytes to bbolt cache", "key", string(cacheKey), "bytes", len(encodedBytes))
							return b.Put(cacheKey, encodedBytes)
						})
						if saveErr == nil {
							logger.Debug("Saved analysis results to bbolt cache", "key", string(cacheKey), "duration", time.Since(saveStart))
						} else {
							logger.Warn("Failed to write to bbolt cache", "key", string(cacheKey), "error", saveErr)
							addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheWrite, saveErr), logger) // Use ErrCacheWrite
						}
					} else {
						logger.Warn("Failed to gob-encode cache entry", "error", entryEncodeErr)
						addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr), logger) // Use ErrCacheEncode from errors.go
					}
				} else {
					logger.Warn("Failed to gob-encode analysis data", "error", encodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr), logger) // Use ErrCacheEncode
				}
			} else {
				logger.Warn("Failed to calculate input hashes for cache save", "error", hashErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger) // Use ErrCacheHash from errors.go
			}
		} else if a.db != nil {
			logger.Debug("Skipping bbolt cache save", "key", string(cacheKey), "load_errors", len(loadErrors), "preamble_empty", info.PromptPreamble == "")
		}
	} // End if !cacheHit

	if cacheHit {
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
	logger := stdslog.Default().With("dir", dir)
	a.mu.Lock()
	db := a.db
	a.mu.Unlock()

	if db == nil {
		logger.Debug("Bbolt cache invalidation skipped: DB is nil.")
		return nil
	}
	goModHash := calculateGoModHash(dir) // Helper from utils.go
	cacheKey := []byte(dir + "::" + goModHash)
	logger.Info("Invalidating bbolt cache entry", "key", string(cacheKey))
	return deleteCacheEntryByKey(db, cacheKey, logger) // Helper from utils.go
}

// InvalidateMemoryCacheForURI clears relevant entries from the ristretto memory cache.
func (a *GoPackagesAnalyzer) InvalidateMemoryCacheForURI(uri string, version int) error {
	logger := stdslog.Default().With("uri", uri, "version", version)
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

// MemoryCacheEnabled returns true if the Ristretto cache is initialized and available.
func (a *GoPackagesAnalyzer) MemoryCacheEnabled() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.memoryCache != nil
}

// GetMemoryCacheMetrics returns the performance metrics collected by Ristretto.
func (a *GoPackagesAnalyzer) GetMemoryCacheMetrics() *ristretto.Metrics {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.memoryCache != nil {
		return a.memoryCache.Metrics
	}
	return nil
}

// --- Default Prompt Formatter ---

// templateFormatter implements the PromptFormatter interface using standard Go templates.
type templateFormatter struct{}

// newTemplateFormatter creates a new instance of the default formatter.
func newTemplateFormatter() *templateFormatter { return &templateFormatter{} }

// FormatPrompt combines the context preamble and code snippet into the final LLM prompt string.
// Cycle 1: Uses Config and SnippetContext from types.go
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2
	logger := stdslog.Default()

	// Truncate context preamble
	if len(contextPreamble) > maxPreambleLen {
		logger.Warn("Truncating context preamble", "original_length", len(contextPreamble), "max_length", maxPreambleLen)
		marker := "... (context truncated)\n"
		startByte := len(contextPreamble) - maxPreambleLen + len(marker)
		if startByte < 0 {
			startByte = 0
		}
		contextPreamble = marker + contextPreamble[startByte:]
	}

	// Format based on FIM setting
	if config.UseFim {
		template = config.FimTemplate
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix

		// Truncate FIM prefix
		if len(prefix) > maxFIMPartLen {
			logger.Warn("Truncating FIM prefix", "original_length", len(prefix), "max_length", maxFIMPartLen)
			marker := "...(prefix truncated)"
			startByte := len(prefix) - maxFIMPartLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			prefix = marker + prefix[startByte:]
		}
		// Truncate FIM suffix
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
		// Truncate snippet (prefix)
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

// DeepCompleter orchestrates the code analysis, prompt formatting, and LLM interaction.
type DeepCompleter struct {
	client    LLMClient       // Interface for interacting with the LLM backend.
	analyzer  Analyzer        // Interface for performing code analysis.
	formatter PromptFormatter // Interface for formatting the LLM prompt.
	config    Config          // Current active configuration (Type from types.go)
	configMu  sync.RWMutex    // Mutex to protect concurrent access to config.
	logger    *stdslog.Logger // Logger instance
}

// NewDeepCompleter creates a new DeepCompleter service instance.
// It loads the configuration using LoadConfig and initializes default components.
// Returns the completer and a potential non-fatal ErrConfig if loading had issues.
func NewDeepCompleter(logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default()
	}

	// Load configuration (handles defaults, merging, validation, writing defaults)
	// Uses ErrConfig from errors.go
	cfg, configErr := LoadConfig(logger)
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		logger.Error("Fatal error during initial config load", "error", configErr)
		return nil, configErr
	}
	// Use Validate method from Config type (defined in types.go)
	if err := cfg.Validate(logger); err != nil {
		if errors.Is(err, ErrInvalidConfig) { // Use ErrInvalidConfig from errors.go
			logger.Error("Loaded/default config is invalid", "error", err)
			return nil, fmt.Errorf("initial config validation failed: %w", err)
		}
		logger.Warn("Initial config validation reported issues", "error", err)
	}

	analyzer := NewGoPackagesAnalyzer(logger)
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    cfg, // Use Config type from types.go
		logger:    logger,
	}

	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with a specific,
// provided configuration, bypassing the standard loading process.
func NewDeepCompleterWithConfig(config Config, logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
	// Ensure internal templates are set if missing
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate // Constant from types.go
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate // Constant from types.go
	}
	// Validate the provided configuration
	// Use Validate method from Config type (defined in types.go)
	if err := config.Validate(logger); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err) // Use ErrInvalidConfig from errors.go
	}

	analyzer := NewGoPackagesAnalyzer(logger)
	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    config, // Use Config type from types.go
		logger:    logger,
	}, nil
}

// Close cleans up resources used by the DeepCompleter.
func (dc *DeepCompleter) Close() error {
	if dc.analyzer != nil {
		return dc.analyzer.Close()
	}
	return nil
}

// UpdateConfig atomically updates the completer's configuration after validating the new config.
func (dc *DeepCompleter) UpdateConfig(newConfig Config) error {
	// Ensure internal templates are set
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate // Constant from types.go
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate // Constant from types.go
	}
	// Validate the incoming configuration
	// Use Validate method from Config type (defined in types.go)
	if err := newConfig.Validate(dc.logger); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidConfig, err) // Use ErrInvalidConfig from errors.go
	}

	dc.configMu.Lock()
	defer dc.configMu.Unlock()
	dc.config = newConfig

	// Log the updated configuration values (Expanded)
	dc.logger.Info("DeepCompleter configuration updated",
		stdslog.String("ollama_url", newConfig.OllamaURL),
		stdslog.String("model", newConfig.Model),
		stdslog.Int("max_tokens", newConfig.MaxTokens),
		stdslog.Any("stop", newConfig.Stop), // Use Any for slice logging
		stdslog.Float64("temperature", newConfig.Temperature),
		stdslog.String("log_level", newConfig.LogLevel),
		stdslog.Bool("use_ast", newConfig.UseAst),
		stdslog.Bool("use_fim", newConfig.UseFim),
		stdslog.Int("max_preamble_len", newConfig.MaxPreambleLen),
		stdslog.Int("max_snippet_len", newConfig.MaxSnippetLen),
	)
	// TODO: Potentially trigger actions based on config changes (e.g., update logger level)
	return nil
}

// GetCurrentConfig returns a thread-safe copy of the current configuration.
func (dc *DeepCompleter) GetCurrentConfig() Config {
	dc.configMu.RLock()
	defer dc.configMu.RUnlock()
	cfgCopy := dc.config // Uses Config type from types.go
	if cfgCopy.Stop != nil {
		stopsCopy := make([]string, len(cfgCopy.Stop))
		copy(stopsCopy, cfgCopy.Stop)
		cfgCopy.Stop = stopsCopy
	}
	return cfgCopy
}

// InvalidateAnalyzerCache provides external access to invalidate the analyzer's disk cache.
func (dc *DeepCompleter) InvalidateAnalyzerCache(dir string) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	return dc.analyzer.InvalidateCache(dir)
}

// InvalidateMemoryCacheForURI provides external access to invalidate the analyzer's memory cache.
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

// GetCompletion provides basic code completion for a given snippet without file context analysis.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	logger := dc.logger.With("operation", "GetCompletion")
	logger.Info("Handling basic completion request")
	currentConfig := dc.GetCurrentConfig()

	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet} // Use SnippetContext from types.go

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
		streamErr := streamCompletion(streamCtx, reader, &buffer) // Helper from utils.go
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr) // Use ErrStreamProcessing from errors.go
		}
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // Helper from utils.go, uses constants from types.go
	if err != nil {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
		// Use ErrOllamaUnavailable, ErrStreamProcessing from errors.go
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) {
			logger.Error("Ollama unavailable for basic completion after retries", "error", err)
			return "", fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error for basic completion after retries", "error", err)
			return "", err
		}
		logger.Error("Failed to get basic completion after retries", "error", err)
		return "", fmt.Errorf("failed to get basic completion after %d retries: %w", maxRetries, err) // Constant from types.go
	}

	logger.Info("Basic completion successful")
	return strings.TrimSpace(buffer.String()), nil
}

// GetCompletionStreamFromFile provides context-aware code completion by analyzing the
// specified file and position, then streams the LLM response to the provided writer.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, absFilename string, version int, line, col int, w io.Writer) error {
	logger := dc.logger.With("operation", "GetCompletionStreamFromFile", "path", absFilename, "version", version, "line", line, "col", col)
	currentConfig := dc.GetCurrentConfig()
	var contextPreamble string = "// Basic file context only."
	var analysisInfo *AstContextInfo // Use AstContextInfo from types.go
	var analysisErr error

	if currentConfig.UseAst {
		logger.Info("Analyzing context (or checking cache)")
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second)
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, absFilename, version, line, col)
		cancelAnalysis()

		// Use ErrAnalysisFailed from errors.go
		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			logger.Error("Fatal error during analysis/cache check", "error", analysisErr)
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		if analysisErr != nil {
			logger.Warn("Non-fatal error during analysis/cache check", "error", analysisErr)
			if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
				contextPreamble = analysisInfo.PromptPreamble + fmt.Sprintf("\n// Warning: Context analysis completed with errors: %v\n", analysisErr)
			} else {
				contextPreamble += fmt.Sprintf("\n// Warning: Context analysis completed with errors: %v\n", analysisErr)
			}
		} else if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else {
			contextPreamble += "\n// Warning: Context analysis returned no specific context preamble.\n"
		}
	} else {
		logger.Info("AST analysis disabled by config.")
	}

	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col) // Helper from utils.go, returns SnippetContext from types.go
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
		streamErr := streamCompletion(apiCtx, reader, w) // Helper from utils.go
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr) // Use ErrStreamProcessing from errors.go
		}
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // Helper from utils.go, uses constants from types.go
	if err != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		// Use ErrOllamaUnavailable, ErrStreamProcessing from errors.go
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) {
			logger.Error("Ollama unavailable for stream after retries", "error", err)
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error for stream after retries", "error", err)
			return err
		}
		logger.Error("Failed to get completion stream after retries", "error", err)
		return fmt.Errorf("failed to get completion stream after %d retries: %w", maxRetries, err) // Constant from types.go
	}

	if analysisErr != nil {
		logger.Warn("Completion stream successful, but context analysis encountered non-fatal errors", "analysis_error", analysisErr)
	} else {
		logger.Info("Completion stream successful")
	}
	return nil
}
