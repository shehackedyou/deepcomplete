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
	"log/slog"
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
	"golang.org/x/tools/go/packages"
)

// =============================================================================
// Global Variables (Package Level)
// =============================================================================

var (
	cacheBucketName = []byte("AnalysisCache") // Name of the bbolt bucket for caching.
)

// Core type definitions are in deepcomplete_types.go.
// Exported error variables are in deepcomplete_errors.go.

// =============================================================================
// Interfaces for Components
// =============================================================================

// LLMClient defines the interface for interacting with the language model backend.
type LLMClient interface {
	// GenerateStream sends a prompt to the LLM and returns a stream of generated text.
	GenerateStream(ctx context.Context, prompt string, config Config, logger *stdslog.Logger) (io.ReadCloser, error)
	// CheckAvailability checks if the LLM backend is reachable.
	CheckAvailability(ctx context.Context, config Config, logger *stdslog.Logger) error
}

// Analyzer defines the interface for code analysis.
type Analyzer interface {
	// Analyze performs static analysis on the given file at the specified position.
	// Returns AstContextInfo and potentially ErrAnalysisFailed if non-fatal errors occurred.
	// TODO: Deprecate this in favor of more specific methods.
	Analyze(ctx context.Context, filename string, version int, line, col int) (*AstContextInfo, error)

	// GetIdentifierInfo attempts to find information about the identifier at the given position.
	GetIdentifierInfo(ctx context.Context, filename string, version int, line, col int) (*IdentifierInfo, error)

	// GetEnclosingContext retrieves information about the function/method enclosing the position.
	GetEnclosingContext(ctx context.Context, filename string, version int, line, col int) (*EnclosingContextInfo, error)

	// GetScopeInfo retrieves variables/types available in the scope at the position.
	GetScopeInfo(ctx context.Context, filename string, version int, line, col int) (*ScopeInfo, error)

	// GetRelevantComments retrieves comments near the cursor position.
	GetRelevantComments(ctx context.Context, filename string, version int, line, col int) ([]string, error)

	// GetPromptPreamble constructs the context preamble for the LLM prompt.
	GetPromptPreamble(ctx context.Context, filename string, version int, line, col int) (string, error)

	// GetMemoryCache retrieves an item from the memory cache.
	GetMemoryCache(key string) (any, bool)
	// SetMemoryCache adds an item to the memory cache.
	SetMemoryCache(key string, value any, cost int64, ttl time.Duration) bool
	// UpdateConfig updates the analyzer's internal config reference (e.g., for cache TTL).
	UpdateConfig(cfg Config)

	// Close cleans up any resources used by the analyzer.
	Close() error
	// InvalidateCache removes cached data related to a specific directory.
	InvalidateCache(dir string) error
	// InvalidateMemoryCacheForURI removes memory-cached data for a specific file URI.
	InvalidateMemoryCacheForURI(uri string, version int) error
	// MemoryCacheEnabled returns true if the in-memory cache is configured and active.
	MemoryCacheEnabled() bool
	// GetMemoryCacheMetrics returns performance metrics for the in-memory cache.
	GetMemoryCacheMetrics() *ristretto.Metrics
}

// PromptFormatter defines the interface for constructing the final prompt sent to the LLM.
type PromptFormatter interface {
	// FormatPrompt combines the analysis preamble and code snippet context into the final prompt string.
	FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config, logger *stdslog.Logger) string
}

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadConfig loads configuration from standard locations, merges with defaults,
// validates, and attempts to write a default config if needed.
func LoadConfig(logger *stdslog.Logger) (Config, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
	cfg := getDefaultConfig()
	var loadedFromFile bool
	var loadErrors []error
	var configParseError error

	primaryPath, secondaryPath, pathErr := GetConfigPaths(logger)
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		logger.Warn("Could not determine config paths, using defaults", "error", pathErr)
	}

	if primaryPath != "" {
		logger.Debug("Attempting to load config", "path", primaryPath)
		loaded, loadErr := LoadAndMergeConfig(primaryPath, &cfg, logger)
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

	primaryNotFoundOrFailed := !loadedFromFile || configParseError != nil
	if primaryNotFoundOrFailed && secondaryPath != "" && secondaryPath != primaryPath {
		logger.Debug("Attempting to load config from secondary path", "path", secondaryPath)
		loaded, loadErr := LoadAndMergeConfig(secondaryPath, &cfg, logger)
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

	loadSucceeded := loadedFromFile && configParseError == nil
	if !loadSucceeded {
		writePath := primaryPath
		if writePath == "" {
			writePath = secondaryPath
		}

		if writePath != "" {
			if configParseError != nil {
				logger.Warn("Existing config file failed to parse. Attempting to write default.", "path", writePath, "error", configParseError)
			} else {
				logger.Info("No valid config file found. Attempting to write default.", "path", writePath)
			}
			if err := WriteDefaultConfig(writePath, getDefaultConfig(), logger); err != nil {
				logger.Warn("Failed to write default config", "path", writePath, "error", err)
				loadErrors = append(loadErrors, fmt.Errorf("writing default config failed: %w", err))
			}
		} else {
			logger.Warn("Cannot determine path to write default config.")
			loadErrors = append(loadErrors, errors.New("cannot determine default config path"))
		}
		cfg = getDefaultConfig() // Fallback to defaults if loading or writing default fails
		logger.Info("Using default configuration values.")
	}

	finalCfg := cfg
	if err := finalCfg.Validate(logger); err != nil {
		logger.Error("Final configuration is invalid, falling back to pure defaults.", "error", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		pureDefault := getDefaultConfig()
		if valErr := pureDefault.Validate(logger); valErr != nil {
			// This should ideally not happen if getDefaultConfig() is correct.
			logger.Error("FATAL: Default config definition is invalid", "error", valErr)
			return pureDefault, fmt.Errorf("default config definition is invalid: %w", valErr)
		}
		finalCfg = pureDefault
	}

	if len(loadErrors) > 0 {
		return finalCfg, fmt.Errorf("%w: %w", ErrConfig, errors.Join(loadErrors...))
	}
	return finalCfg, nil
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
				TLSHandshakeTimeout:   10 * time.Second,
				MaxIdleConns:          10,
				IdleConnTimeout:       30 * time.Second,
				ResponseHeaderTimeout: 20 * time.Second, // Timeout for receiving response headers
			},
		},
	}
}

// CheckAvailability sends a simple request to the Ollama base URL to check reachability.
func (c *httpOllamaClient) CheckAvailability(ctx context.Context, config Config, logger *stdslog.Logger) error {
	if logger == nil {
		logger = stdslog.Default()
	}
	checkLogger := logger.With("operation", "CheckAvailability", "url", config.OllamaURL)
	checkLogger.Debug("Checking Ollama availability")

	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second) // Short timeout for availability check
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, config.OllamaURL, nil)
	if err != nil {
		checkLogger.Error("Failed to create availability check request", "error", err)
		return fmt.Errorf("%w: failed to create check request: %w", ErrOllamaUnavailable, err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			checkLogger.Error("Timeout checking Ollama availability", "error", err)
		} else {
			checkLogger.Error("Failed to connect to Ollama for availability check", "error", err)
		}
		return fmt.Errorf("%w: availability check failed: %w", ErrOllamaUnavailable, err)
	}
	defer resp.Body.Close()

	checkLogger.Debug("Ollama availability check successful", "status", resp.StatusCode)
	return nil
}

// GenerateStream sends a request to Ollama's /api/generate endpoint and returns the streaming response body.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config, logger *stdslog.Logger) (io.ReadCloser, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
	opLogger := logger.With("operation", "GenerateStream", "model", config.Model)

	base := strings.TrimSuffix(config.OllamaURL, "/")
	endpointURL := base + "/api/generate"
	u, err := url.Parse(endpointURL)
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL '%s': %w", endpointURL, err)
	}

	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true, // Ensure streaming is enabled
		"options": map[string]interface{}{ // Common options
			"temperature": config.Temperature,
			"num_ctx":     4096, // Context window size for the model
			"top_p":       0.9,  // Standard top_p value
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
	req.Header.Set("Accept", "application/x-ndjson") // Expect newline-delimited JSON stream

	opLogger.Debug("Sending generate request to Ollama", "url", endpointURL)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Handle context cancellation/deadline specifically
		if errors.Is(err, context.Canceled) {
			opLogger.Warn("Ollama generate request context cancelled", "url", endpointURL)
			return nil, context.Canceled
		}
		if errors.Is(err, context.DeadlineExceeded) {
			opLogger.Error("Ollama generate request context deadline exceeded", "url", endpointURL, "timeout", c.httpClient.Timeout)
			return nil, fmt.Errorf("%w: context deadline exceeded: %w", ErrOllamaUnavailable, context.DeadlineExceeded)
		}

		// Handle network errors
		var netErr net.Error
		if errors.As(err, &netErr) {
			if netErr.Timeout() {
				opLogger.Error("Network timeout during Ollama generate request", "host", u.Host, "error", netErr)
				return nil, fmt.Errorf("%w: network timeout: %w", ErrOllamaUnavailable, netErr)
			}
			// Check for connection refused specifically
			if opErr, ok := netErr.(*net.OpError); ok && opErr.Op == "dial" {
				opLogger.Error("Connection refused or network error during Ollama generate request", "host", u.Host, "error", opErr)
				return nil, fmt.Errorf("%w: connection failed: %w", ErrOllamaUnavailable, opErr)
			}
		}

		opLogger.Error("HTTP request to Ollama generate failed", "url", endpointURL, "error", err)
		return nil, fmt.Errorf("%w: http request failed: %w", ErrOllamaUnavailable, err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(failed to read error response body)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			// Attempt to parse Ollama's specific error structure
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error
			}
		}
		apiErr := &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		opLogger.Error("Ollama API returned non-OK status", "status", resp.Status, "response_body", bodyString)
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, apiErr)
	}

	return resp.Body, nil
}

// --- Default Analyzer ---

// GoPackagesAnalyzer implements the Analyzer interface using go/packages and bbolt/ristretto caching.
type GoPackagesAnalyzer struct {
	db          *bbolt.DB        // Persistent disk cache (bbolt)
	memoryCache *ristretto.Cache // In-memory cache (ristretto)
	mu          sync.RWMutex     // Protects access to db/memoryCache handles AND config
	logger      *stdslog.Logger  // Stored logger instance
	config      Config           // Store config to access TTL easily
}

// NewGoPackagesAnalyzer initializes the analyzer, including setting up bbolt and ristretto caches.
func NewGoPackagesAnalyzer(logger *stdslog.Logger, initialConfig Config) *GoPackagesAnalyzer {
	if logger == nil {
		logger = stdslog.Default()
	}
	analyzerLogger := logger.With("component", "GoPackagesAnalyzer")

	dbPath := ""
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil { // Ensure directory exists with appropriate permissions
			dbPath = filepath.Join(dbDir, "analysis_cache.db")
		} else {
			analyzerLogger.Warn("Could not create bbolt cache directory, disk caching disabled.", "path", dbDir, "error", err)
		}
	} else {
		analyzerLogger.Warn("Could not determine user cache directory, disk caching disabled.", "error", err)
	}

	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second} // Options for opening the DB
		db, err = bbolt.Open(dbPath, 0600, opts)         // Open with restricted file permissions
		if err != nil {
			analyzerLogger.Warn("Failed to open bbolt cache file, disk caching disabled.", "path", dbPath, "error", err)
			db = nil // Ensure db is nil if open fails
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
				analyzerLogger.Warn("Failed to ensure bbolt bucket exists, disk caching disabled.", "error", err)
				db.Close() // Close DB if bucket creation fails
				db = nil
			} else {
				analyzerLogger.Info("Using bbolt disk cache", "path", dbPath, "schema_version", cacheSchemaVersion)
			}
		}
	}

	// Initialize Ristretto in-memory cache
	memCache, cacheErr := ristretto.NewCache(&ristretto.Config{
		NumCounters: 1e7,     // Number of keys to track frequency of (10M)
		MaxCost:     1 << 30, // Maximum cost of cache (1GB)
		BufferItems: 64,      // Number of keys per Get buffer.
		Metrics:     true,    // Enable metrics for monitoring
	})
	if cacheErr != nil {
		analyzerLogger.Warn("Failed to create ristretto memory cache, in-memory caching disabled.", "error", cacheErr)
		memCache = nil
	} else {
		analyzerLogger.Info("Initialized ristretto in-memory cache", "max_cost", "1GB")
	}

	return &GoPackagesAnalyzer{
		db:          db,
		memoryCache: memCache,
		logger:      analyzerLogger,
		config:      initialConfig, // Store initial config
	}
}

// UpdateConfig updates the analyzer's internal config reference.
func (a *GoPackagesAnalyzer) UpdateConfig(cfg Config) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.config = cfg
	a.logger.Info("Analyzer configuration updated", "new_ttl_seconds", cfg.MemoryCacheTTLSeconds)
}

// getConfig provides thread-safe access to the analyzer's config.
func (a *GoPackagesAnalyzer) getConfig() Config {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.config
}

// Close cleans up resources used by the analyzer.
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock() // Use write lock for closing
	defer a.mu.Unlock()
	var closeErrors []error
	logger := a.logger // Use the analyzer's logger

	if a.db != nil {
		logger.Info("Closing bbolt cache database.")
		if err := a.db.Close(); err != nil {
			logger.Error("Error closing bbolt database", "error", err)
			closeErrors = append(closeErrors, fmt.Errorf("bbolt close failed: %w", err))
		}
		a.db = nil // Set to nil after closing
	}
	if a.memoryCache != nil {
		logger.Info("Closing ristretto memory cache.")
		a.memoryCache.Close()
		a.memoryCache = nil // Set to nil after closing
	}

	if len(closeErrors) > 0 {
		return errors.Join(closeErrors...)
	}
	return nil
}

// loadPackageForAnalysis handles the core logic of loading package data.
// This is an internal helper for the Analyzer methods.
func (a *GoPackagesAnalyzer) loadPackageForAnalysis(ctx context.Context, absFilename string, fset *token.FileSet, logger *slog.Logger) (*packages.Package, *ast.File, *token.File, []Diagnostic, []error) {
	// Delegate to the standalone helper function in helpers_loader.go
	return loadPackageAndFile(ctx, absFilename, fset, logger)
}

// Analyze performs static analysis on the given file and position, utilizing caching.
// Returns the populated AstContextInfo and any fatal error encountered.
// Non-fatal errors are stored in AstContextInfo.AnalysisErrors.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, absFilename string, version int, line, col int) (info *AstContextInfo, analysisErr error) {
	analysisLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "Analyze")

	info = &AstContextInfo{
		FilePath:         absFilename,
		Version:          version,
		VariablesInScope: make(map[string]types.Object), // Initialize maps
		AnalysisErrors:   make([]error, 0),
		Diagnostics:      make([]Diagnostic, 0),
		CallArgIndex:     -1, // Default to -1 (no active call arg)
	}

	// Panic recovery for the Analyze method
	defer func() {
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			analysisLogger.Error("Panic recovered during Analyze", "error", r, "stack", string(debug.Stack()))
			addAnalysisError(info, panicErr, analysisLogger) // Add to non-fatal errors
			if analysisErr == nil {
				analysisErr = panicErr // If no other fatal error, this becomes the fatal one
			} else {
				// If there was already a fatal error, append panic info
				analysisErr = fmt.Errorf("panic (%v) occurred after error: %w", r, analysisErr)
			}
		}
		// If non-fatal errors occurred and no fatal error is set, wrap them in ErrAnalysisFailed
		if len(info.AnalysisErrors) > 0 && analysisErr == nil {
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr)
		} else if len(info.AnalysisErrors) > 0 && analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			// If a fatal error occurred AND non-fatal errors, include non-fatal in the message
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w (additional analysis issues: %w)", analysisErr, finalErr)
		}
	}()

	analysisLogger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)
	goModHash := calculateGoModHash(dir, analysisLogger)
	cacheKey := []byte(dir + "::" + goModHash) // Key for bbolt cache
	cacheHit := false                          // Flag for bbolt cache hit
	var cachedEntry *CachedAnalysisEntry
	var loadDuration, stepsDuration, preambleDuration time.Duration

	// --- Bbolt Cache Check (for preamble) ---
	if a.db != nil {
		readStart := time.Now()
		dbViewErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				return nil // Bucket doesn't exist, treat as miss
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				return nil // Key not found, treat as miss
			}

			var decoded CachedAnalysisEntry
			if err := gob.NewDecoder(bytes.NewReader(valBytes)).Decode(&decoded); err != nil {
				return fmt.Errorf("%w: failed to decode cache entry: %w", ErrCacheDecode, err)
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				analysisLogger.Warn("Bbolt cache data has old schema version. Ignoring.", "key", string(cacheKey), "cached_version", decoded.SchemaVersion, "expected_version", cacheSchemaVersion)
				return nil // Schema mismatch, treat as miss and invalidate later
			}
			cachedEntry = &decoded
			return nil
		})
		if dbViewErr != nil {
			analysisLogger.Warn("Error reading or decoding from bbolt cache.", "error", dbViewErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbViewErr), analysisLogger)
			if errors.Is(dbViewErr, ErrCacheDecode) { // Specific error for decode failure
				go deleteCacheEntryByKey(a.db, cacheKey, analysisLogger.With("reason", "decode_failure"))
			}
			cachedEntry = nil // Ensure it's nil on error
		}
		analysisLogger.Debug("Bbolt cache read attempt finished", "duration", time.Since(readStart))

		if cachedEntry != nil {
			validationStart := time.Now()
			currentHashes, hashErr := calculateInputHashes(dir, nil, analysisLogger) // Pass nil for pkg as we don't have it yet
			if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes, analysisLogger) {
				decodeStart := time.Now()
				var analysisData CachedAnalysisData
				if decodeErr := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob)).Decode(&analysisData); decodeErr == nil {
					info.PackageName = analysisData.PackageName
					info.PromptPreamble = analysisData.PromptPreamble
					// cacheHit = true // This was for preamble, but Analyze does more.
					// For Analyze, a bbolt hit for preamble doesn't mean full analysis is cached.
					loadDuration = time.Since(decodeStart)
					analysisLogger.Debug("Preamble successfully decoded from bbolt cache.", "duration", loadDuration)
					// Set cacheHit to false to force re-analysis for other parts if Analyze is called.
					// Specific getters will handle their own caching.
					cacheHit = false
				} else {
					analysisLogger.Warn("Failed to gob-decode cached analysis data. Treating as miss.", "error", decodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheDecode, decodeErr), analysisLogger)
					go deleteCacheEntryByKey(a.db, cacheKey, analysisLogger.With("reason", "analysis_decode_failure"))
					cacheHit = false
				}
			} else {
				analysisLogger.Debug("Bbolt cache INVALID (hash mismatch or error). Treating as miss.", "key", string(cacheKey), "hash_error", hashErr)
				go deleteCacheEntryByKey(a.db, cacheKey, analysisLogger.With("reason", "hash_mismatch"))
				if hashErr != nil {
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), analysisLogger)
				}
				cacheHit = false
			}
			analysisLogger.Debug("Bbolt cache validation/decode finished", "duration", time.Since(validationStart))
		}
	} else {
		analysisLogger.Debug("Bbolt cache disabled (db handle is nil).")
	}
	// --- End Bbolt Cache Check ---

	// --- Load Package Info (Extracted Logic) ---
	loadStart := time.Now()
	fset := token.NewFileSet()
	info.TargetFileSet = fset // Store FileSet in info
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, analysisLogger)
	info.Diagnostics = append(info.Diagnostics, loadDiagnostics...) // Append diagnostics from loading
	loadDuration = time.Since(loadStart)
	analysisLogger.Debug("Package loading completed", "duration", loadDuration)
	for _, loadErr := range loadErrors {
		addAnalysisError(info, loadErr, analysisLogger) // Add non-fatal load errors
	}
	if len(loadErrors) > 0 && targetPkg == nil { // Critical load error if targetPkg is nil
		analysisLogger.Error("Critical package load error occurred.")
		// Defer will handle setting analysisErr based on info.AnalysisErrors
		return info, analysisErr // Return info and allow defer to set analysisErr
	}
	info.TargetPackage = targetPkg
	info.TargetAstFile = targetFileAST
	// --- End Load Package Info ---

	// --- Perform Analysis Steps (using loaded info) ---
	// This section is a candidate for major refactoring. The specific Get* methods
	// should be the primary way to obtain detailed context.
	stepsStart := time.Now()
	if targetFile != nil { // Ensure targetFile (token.File) is valid
		// The call to performAnalysisSteps is becoming redundant as specific getters are enhanced.
		// For now, it's kept to ensure existing behavior of Analyze method.
		// TODO: Refactor Analyze to directly use specific getters or a more streamlined internal process.
		analyzeStepErr := performAnalysisSteps(ctx, targetFile, targetFileAST, targetPkg, fset, line, col, a, info, analysisLogger)
		if analyzeStepErr != nil {
			// performAnalysisSteps now returns error only for fatal issues like invalid cursor.
			analysisLogger.Error("Fatal error during performAnalysisSteps", "error", analyzeStepErr)
			analysisErr = analyzeStepErr // Set the fatal error
			return info, analysisErr     // Return immediately
		}
		// Non-fatal errors from performAnalysisSteps are added to info.AnalysisErrors within that function.
	} else {
		if len(loadErrors) == 0 { // Only log if no previous load error explains why targetFile is nil
			addAnalysisError(info, errors.New("cannot perform analysis steps: target token.File is nil"), analysisLogger)
		}
		// Attempt to gather package scope even if file-specific analysis fails
		// This would typically be handled by GetScopeInfo if called directly.
	}
	stepsDuration = time.Since(stepsStart)
	analysisLogger.Debug("Analysis steps completed", "duration", stepsDuration)
	// --- End Analysis Steps ---

	// --- Construct Preamble (if not loaded from bbolt cache) ---
	if info.PromptPreamble == "" { // Only generate if not already populated (e.g., by bbolt cache)
		preambleStart := time.Now()
		preamble, preambleErr := a.GetPromptPreamble(ctx, absFilename, version, line, col)
		if preambleErr != nil {
			analysisLogger.Error("Error getting prompt preamble", "error", preambleErr)
			addAnalysisError(info, fmt.Errorf("failed to get prompt preamble: %w", preambleErr), analysisLogger)
			// Use a basic fallback preamble on error
			info.PromptPreamble = fmt.Sprintf("// File: %s\n// Package: %s\n// ERROR: Could not generate detailed preamble.\n", filepath.Base(absFilename), info.PackageName)
		} else {
			info.PromptPreamble = preamble
		}
		preambleDuration = time.Since(preambleStart)
		analysisLogger.Debug("Preamble construction/retrieval completed", "duration", preambleDuration)
	} else {
		analysisLogger.Debug("Skipping preamble construction as it was populated (e.g., from bbolt cache).")
	}
	// --- End Construct Preamble ---

	// --- Bbolt Cache Write (if preamble was generated and no critical load errors) ---
	// Only save if db is available, preamble was generated (not from cache hit for preamble itself),
	// and no critical package loading errors occurred.
	shouldSave := a.db != nil && info.PromptPreamble != "" && !cacheHit && len(loadErrors) == 0
	if shouldSave {
		analysisLogger.Debug("Attempting to save analysis results to bbolt cache.", "key", string(cacheKey))
		saveStart := time.Now()
		// Pass targetPkg to calculateInputHashes if available
		inputHashes, hashErr := calculateInputHashes(dir, targetPkg, analysisLogger)
		if hashErr == nil {
			// Cache only essential data for preamble reconstruction
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
						if b == nil { // Should not happen if initialized correctly
							return fmt.Errorf("%w: cache bucket %s disappeared", ErrCacheWrite, string(cacheBucketName))
						}
						return b.Put(cacheKey, encodedBytes)
					})
					if saveErr == nil {
						analysisLogger.Debug("Saved analysis results to bbolt cache", "key", string(cacheKey), "duration", time.Since(saveStart))
					} else {
						analysisLogger.Warn("Failed to write to bbolt cache", "key", string(cacheKey), "error", saveErr)
						addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheWrite, saveErr), analysisLogger)
					}
				} else {
					analysisLogger.Warn("Failed to gob-encode cache entry for bbolt", "error", entryEncodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr), analysisLogger)
				}
			} else {
				analysisLogger.Warn("Failed to gob-encode analysis data for bbolt", "error", encodeErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr), analysisLogger)
			}
		} else {
			analysisLogger.Warn("Failed to calculate input hashes for bbolt cache save", "error", hashErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), analysisLogger)
		}
	} else if a.db != nil {
		analysisLogger.Debug("Skipping bbolt cache save", "key", string(cacheKey), "cache_hit_for_preamble", cacheHit, "load_errors_present", len(loadErrors) > 0, "preamble_empty", info.PromptPreamble == "")
	}
	// --- End Bbolt Cache Write ---

	analysisLogger.Info("Context analysis finished", "load_duration", loadDuration, "steps_duration", stepsDuration, "preamble_duration", preambleDuration)
	analysisLogger.Debug("Final Context Preamble generated", "length", len(info.PromptPreamble))
	analysisLogger.Debug("Final Diagnostics collected", "count", len(info.Diagnostics))

	// analysisErr is set by the defer func based on info.AnalysisErrors or fatal errors
	return info, analysisErr
}

// GetIdentifierInfo implements the Analyzer interface method.
// It attempts to find detailed information about the identifier at the given cursor position.
func (a *GoPackagesAnalyzer) GetIdentifierInfo(ctx context.Context, absFilename string, version int, line, col int) (*IdentifierInfo, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetIdentifierInfo")
	opLogger.Info("Starting identifier analysis")

	// 1. Load package and file AST
	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)

	// Create a temporary AstContextInfo to hold intermediate analysis results for this operation.
	// This is populated by findEnclosingPath and findContextNodes.
	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		TargetPackage:  targetPkg,
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST,
		AnalysisErrors: make([]error, 0),   // Collect non-fatal errors here
		Diagnostics:    loadDiagnostics[:], // Start with diagnostics from loading
	}

	if targetPkg == nil {
		combinedErr := errors.Join(loadErrors...)
		opLogger.Error("Critical error loading package, cannot get identifier info", "error", combinedErr)
		// Return nil info, and the combined loading error.
		return nil, fmt.Errorf("critical package load error: %w", combinedErr)
	}
	if targetFile == nil || targetFileAST == nil {
		errText := "target file AST or token.File is nil after loading"
		opLogger.Error(errText, "targetFile_nil", targetFile == nil, "targetAST_nil", targetFileAST == nil)
		// Combine with existing load errors if any.
		return nil, errors.Join(append(loadErrors, errors.New(errText))...)
	}
	if targetPkg.TypesInfo == nil {
		// This is a significant issue for identifier resolution.
		opLogger.Warn("Type info (TypesInfo) is nil for the package. Identifier resolution will likely fail or be incomplete.")
		// Add to tempInfo's errors, it will be wrapped later if it's the only error.
		addAnalysisError(tempInfo, errors.New("package type information (TypesInfo) is missing"), opLogger)
	}

	// 2. Calculate cursor position
	cursorPos, posErr := calculateCursorPos(targetFile, line, col, opLogger)
	if posErr != nil {
		// This is a fatal error for this operation.
		return nil, fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		// Should be caught by calculateCursorPos, but double-check.
		return nil, fmt.Errorf("%w: invalid cursor position calculated (Pos: %d)", ErrPositionConversion, cursorPos)
	}
	tempInfo.CursorPos = cursorPos // Store valid cursor position in tempInfo
	opLogger = opLogger.With("cursorPos", cursorPos, "cursorPosStr", fset.PositionFor(cursorPos, true).String())

	// 3. Find enclosing AST path and specific context nodes (identifier, its type, definition)
	// findEnclosingPath and findContextNodes will populate fields in tempInfo.
	path, pathErr := findEnclosingPath(ctx, targetFileAST, cursorPos, tempInfo, opLogger)
	if pathErr != nil {
		// Add to non-fatal errors, as findContextNodes might still yield some results or partial info.
		opLogger.Warn("Error finding enclosing AST path", "error", pathErr)
		addAnalysisError(tempInfo, fmt.Errorf("finding enclosing AST path failed: %w", pathErr), opLogger)
	}
	// findContextNodes populates tempInfo.IdentifierAtCursor, tempInfo.IdentifierObject,
	// tempInfo.IdentifierType, and calls findDefiningNode (which populates tempInfo.IdentifierDefNode).
	findContextNodes(ctx, path, cursorPos, targetPkg, fset, a, tempInfo, opLogger)

	// Combine all non-fatal errors collected so far (from loading, path finding, context node finding).
	analysisProcessError := errors.Join(tempInfo.AnalysisErrors...)
	if analysisProcessError != nil {
		opLogger.Warn("Non-fatal errors occurred during identifier analysis steps", "errors", analysisProcessError)
		// Wrap these non-fatal errors in ErrAnalysisFailed to signal partial success or warnings.
		analysisProcessError = fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisProcessError)
	}

	// 4. Construct IdentifierInfo if an identifier was found
	if tempInfo.IdentifierAtCursor == nil || tempInfo.IdentifierObject == nil {
		opLogger.Debug("No specific identifier object found at cursor position.")
		// Return nil info, but include any non-fatal analysis errors encountered.
		return nil, analysisProcessError
	}

	// Read file content for accurate range conversion for hover/definition, if needed by consumers.
	fileContent, readErr := os.ReadFile(absFilename)
	if readErr != nil {
		opLogger.Error("Failed to read file content for identifier info construction", "error", readErr)
		// This is a more significant issue if content is needed for range conversion.
		// Combine with existing analysis errors.
		finalErr := errors.Join(analysisProcessError, fmt.Errorf("failed to read file content %s: %w", absFilename, readErr))
		return nil, finalErr
	}

	identInfo := &IdentifierInfo{
		Name:      tempInfo.IdentifierObject.Name(),
		Object:    tempInfo.IdentifierObject,
		Type:      tempInfo.IdentifierType,
		DefNode:   tempInfo.IdentifierDefNode,
		FileSet:   fset, // Provide the FileSet for interpreting positions
		Pkg:       targetPkg,
		Content:   fileContent,                 // Provide content for range conversions
		IdentNode: tempInfo.IdentifierAtCursor, // The *ast.Ident node itself
	}

	// Populate definition file position if available
	if defPos := identInfo.Object.Pos(); defPos.IsValid() {
		identInfo.DefFilePos = fset.Position(defPos) // Get token.Position
	}

	opLogger.Info("Identifier info retrieved successfully", "identifier", identInfo.Name)
	// Return the populated IdentifierInfo and any non-fatal analysis errors.
	return identInfo, analysisProcessError
}

// GetEnclosingContext implements the Analyzer interface method.
func (a *GoPackagesAnalyzer) GetEnclosingContext(ctx context.Context, absFilename string, version int, line, col int) (*EnclosingContextInfo, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetEnclosingContext")
	opLogger.Debug("Starting enclosing context analysis")

	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)

	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		TargetPackage:  targetPkg,
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST,
		AnalysisErrors: make([]error, 0),
		Diagnostics:    loadDiagnostics[:],
	}

	if targetPkg == nil {
		return nil, fmt.Errorf("critical package load error: %w", errors.Join(loadErrors...))
	}
	if targetFile == nil || targetFileAST == nil {
		return nil, errors.Join(append(loadErrors, errors.New("target file AST or token.File is nil"))...)
	}

	cursorPos, posErr := calculateCursorPos(targetFile, line, col, opLogger)
	if posErr != nil {
		return nil, fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
	}
	tempInfo.CursorPos = cursorPos

	// Use memory cache for path finding
	cacheKey := generateCacheKey("astPath", tempInfo)
	computePathFn := func() ([]ast.Node, error) {
		// Pass tempInfo to findEnclosingPath so it can add errors if any occur.
		return findEnclosingPath(ctx, targetFileAST, cursorPos, tempInfo, opLogger)
	}
	path, _, pathErr := withMemoryCache[[]ast.Node](a, cacheKey, 1, a.getConfig().MemoryCacheTTL, computePathFn, opLogger)
	if pathErr != nil {
		// findEnclosingPath itself doesn't add to tempInfo.AnalysisErrors.
		// The error is returned and should be handled here.
		addAnalysisError(tempInfo, fmt.Errorf("finding enclosing AST path failed: %w", pathErr), opLogger)
	}

	// gatherScopeContext populates tempInfo.EnclosingFunc, tempInfo.EnclosingFuncNode,
	// tempInfo.ReceiverType, tempInfo.EnclosingBlock.
	// It also collects errors in tempInfo.AnalysisErrors.
	gatherScopeContext(ctx, path, targetPkg, fset, tempInfo, opLogger)

	enclosingInfo := &EnclosingContextInfo{
		Func:     tempInfo.EnclosingFunc,
		FuncNode: tempInfo.EnclosingFuncNode,
		Receiver: tempInfo.ReceiverType,
		Block:    tempInfo.EnclosingBlock,
	}

	// Combine all non-fatal errors collected.
	analysisProcessError := errors.Join(tempInfo.AnalysisErrors...)
	if analysisProcessError != nil {
		opLogger.Warn("Non-fatal errors during enclosing context analysis", "errors", analysisProcessError)
		// Return the successfully gathered info along with wrapped non-fatal errors.
		return enclosingInfo, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisProcessError)
	}

	return enclosingInfo, nil
}

// GetScopeInfo implements the Analyzer interface method.
func (a *GoPackagesAnalyzer) GetScopeInfo(ctx context.Context, absFilename string, version int, line, col int) (*ScopeInfo, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetScopeInfo")
	opLogger.Debug("Starting scope analysis")

	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)

	// Initialize tempInfo for this operation.
	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		TargetPackage:  targetPkg,
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST,
		AnalysisErrors: make([]error, 0),
		Diagnostics:    loadDiagnostics[:],
	}

	if targetPkg == nil {
		return nil, fmt.Errorf("critical package load error: %w", errors.Join(loadErrors...))
	}
	if targetFile == nil { // targetFileAST can be nil if we only have package scope
		return nil, errors.Join(append(loadErrors, errors.New("target token.File is nil"))...)
	}

	cursorPos, posErr := calculateCursorPos(targetFile, line, col, opLogger)
	if posErr != nil {
		return nil, fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
	}
	tempInfo.CursorPos = cursorPos // Store valid cursor pos

	// Use memory cache for scope extraction.
	// The computeScopeFn will handle finding path and gathering scope.
	cacheKey := generateCacheKey("scopeInfo", tempInfo)
	computeScopeFn := func() (map[string]types.Object, error) {
		// Create a fresh AstContextInfo copy for this computation to avoid shared state issues
		// if the outer tempInfo is used elsewhere or if retries occur.
		// However, for simplicity and if computeScopeFn is not retried by withMemoryCache,
		// directly modifying a local copy or a sub-part of tempInfo might be okay.
		// For now, let's use a local map and collect errors separately.
		localScopeMap := make(map[string]types.Object)
		var computationErrors []error

		// Create a temporary AstContextInfo specifically for this computation,
		// ensuring it has the necessary fields like FilePath, Version, CursorPos, TargetPackage, etc.
		computationTempInfo := &AstContextInfo{
			FilePath:         tempInfo.FilePath,
			Version:          tempInfo.Version,
			CursorPos:        tempInfo.CursorPos,
			TargetPackage:    tempInfo.TargetPackage,
			TargetFileSet:    tempInfo.TargetFileSet,
			TargetAstFile:    tempInfo.TargetAstFile, // Needed for findEnclosingPath
			VariablesInScope: localScopeMap,          // Use local map here
			AnalysisErrors:   make([]error, 0),       // Fresh error slice for this computation
		}

		path, pathErr := findEnclosingPath(ctx, computationTempInfo.TargetAstFile, computationTempInfo.CursorPos, computationTempInfo, opLogger)
		if pathErr != nil {
			addAnalysisError(computationTempInfo, fmt.Errorf("finding enclosing path failed for scope: %w", pathErr), opLogger)
		}

		gatherScopeContext(ctx, path, computationTempInfo.TargetPackage, computationTempInfo.TargetFileSet, computationTempInfo, opLogger)

		// Collect errors from the computation's tempInfo
		computationErrors = append(computationErrors, computationTempInfo.AnalysisErrors...)

		if len(computationErrors) > 0 {
			return localScopeMap, fmt.Errorf("errors during scope computation: %w", errors.Join(computationErrors...))
		}
		return localScopeMap, nil
	}

	scopeMap, cacheHit, scopeErr := withMemoryCache[map[string]types.Object](a, cacheKey, 10, a.getConfig().MemoryCacheTTL, computeScopeFn, opLogger)
	if scopeErr != nil {
		opLogger.Warn("Non-fatal errors during scope extraction", "error", scopeErr)
		// Return the (potentially partially filled) scopeMap along with wrapped non-fatal errors.
		return &ScopeInfo{Variables: scopeMap}, fmt.Errorf("%w: %w", ErrAnalysisFailed, scopeErr)
	}
	if cacheHit {
		opLogger.Debug("Scope info cache hit")
	}

	return &ScopeInfo{Variables: scopeMap}, nil
}

// GetRelevantComments implements the Analyzer interface method.
func (a *GoPackagesAnalyzer) GetRelevantComments(ctx context.Context, absFilename string, version int, line, col int) ([]string, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetRelevantComments")
	opLogger.Debug("Starting comment analysis")

	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)

	// Initialize tempInfo for this operation.
	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		TargetPackage:  targetPkg, // Can be nil if load failed
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST,
		AnalysisErrors: make([]error, 0),
		Diagnostics:    loadDiagnostics[:],
	}

	// Critical check: targetFileAST and targetFile are essential for comment analysis.
	if targetFileAST == nil || targetFile == nil {
		// Combine with existing load errors if any.
		return nil, errors.Join(append(loadErrors, errors.New("target file AST or token.File is nil, cannot get comments"))...)
	}

	cursorPos, posErr := calculateCursorPos(targetFile, line, col, opLogger)
	if posErr != nil {
		return nil, fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
	}
	tempInfo.CursorPos = cursorPos // Store valid cursor pos

	cacheKey := generateCacheKey("comments", tempInfo)
	computeCommentsFn := func() ([]string, error) {
		// Similar to GetScopeInfo, use a computation-local AstContextInfo
		// to manage errors and results for this specific cached operation.
		computationTempInfo := &AstContextInfo{
			FilePath:           tempInfo.FilePath,
			Version:            tempInfo.Version,
			CursorPos:          tempInfo.CursorPos,
			TargetPackage:      tempInfo.TargetPackage, // Pass along for context, though not directly used by findRelevantComments
			TargetFileSet:      tempInfo.TargetFileSet,
			TargetAstFile:      tempInfo.TargetAstFile,
			CommentsNearCursor: nil,              // Initialize for population
			AnalysisErrors:     make([]error, 0), // Fresh error slice
		}

		path, pathErr := findEnclosingPath(ctx, computationTempInfo.TargetAstFile, computationTempInfo.CursorPos, computationTempInfo, opLogger)
		if pathErr != nil {
			addAnalysisError(computationTempInfo, fmt.Errorf("finding enclosing path failed for comments: %w", pathErr), opLogger)
		}

		findRelevantComments(ctx, computationTempInfo.TargetAstFile, path, computationTempInfo.CursorPos, computationTempInfo.TargetFileSet, computationTempInfo, opLogger)

		if len(computationTempInfo.AnalysisErrors) > 0 {
			return computationTempInfo.CommentsNearCursor, fmt.Errorf("errors during comment computation: %w", errors.Join(computationTempInfo.AnalysisErrors...))
		}
		return computationTempInfo.CommentsNearCursor, nil
	}

	comments, cacheHit, commentErr := withMemoryCache[[]string](a, cacheKey, 5, a.getConfig().MemoryCacheTTL, computeCommentsFn, opLogger)
	if commentErr != nil {
		opLogger.Warn("Non-fatal errors during comment extraction", "error", commentErr)
		// Return potentially partial comments along with wrapped non-fatal errors.
		return comments, fmt.Errorf("%w: %w", ErrAnalysisFailed, commentErr)
	}
	if cacheHit {
		opLogger.Debug("Relevant comments cache hit")
	}

	return comments, nil
}

// GetPromptPreamble implements the Analyzer interface method.
// It orchestrates calls to other Get* methods to gather context and then formats it.
func (a *GoPackagesAnalyzer) GetPromptPreamble(ctx context.Context, absFilename string, version int, line, col int) (string, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetPromptPreamble")
	opLogger.Debug("Starting preamble generation")

	// --- Attempt to get preamble from bbolt cache first ---
	// This part remains as bbolt caching is for the final preamble string.
	dir := filepath.Dir(absFilename)
	goModHash := calculateGoModHash(dir, opLogger)
	bboltCacheKey := []byte(dir + "::" + goModHash) // Use a distinct name for bbolt key
	var cachedPreamble string
	var preambleConstructionErrors []error // Collect non-fatal errors during construction

	if a.db != nil {
		dbViewErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				return nil
			}
			valBytes := b.Get(bboltCacheKey)
			if valBytes == nil {
				return nil
			}
			var decoded CachedAnalysisEntry
			if err := gob.NewDecoder(bytes.NewReader(valBytes)).Decode(&decoded); err != nil {
				return fmt.Errorf("%w: %w", ErrCacheDecode, err)
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				opLogger.Warn("Bbolt cache data has old schema version. Ignoring.", "key", string(bboltCacheKey))
				return nil
			}
			// Validate hashes before using cached preamble
			currentHashes, hashErr := calculateInputHashes(dir, nil, opLogger) // Pkg not available yet for hash calc
			if hashErr == nil && decoded.GoModHash == goModHash && compareFileHashes(currentHashes, decoded.InputFileHashes, opLogger) {
				var analysisData CachedAnalysisData
				if decodeErr := gob.NewDecoder(bytes.NewReader(decoded.AnalysisGob)).Decode(&analysisData); decodeErr == nil {
					cachedPreamble = analysisData.PromptPreamble
					opLogger.Debug("Found valid preamble in bbolt cache")
				} else {
					opLogger.Warn("Failed to gob-decode cached analysis data from bbolt. Ignoring.", "error", decodeErr)
					go deleteCacheEntryByKey(a.db, bboltCacheKey, opLogger.With("reason", "analysis_decode_failure"))
				}
			} else {
				opLogger.Debug("Bbolt cache INVALID (hash mismatch or error). Ignoring cached preamble.", "hash_err", hashErr)
				go deleteCacheEntryByKey(a.db, bboltCacheKey, opLogger.With("reason", "hash_mismatch"))
			}
			return nil
		})
		if dbViewErr != nil {
			opLogger.Warn("Error reading or decoding from bbolt cache for preamble.", "error", dbViewErr)
			preambleConstructionErrors = append(preambleConstructionErrors, fmt.Errorf("%w: %w", ErrCacheRead, dbViewErr))
		}
	}

	if cachedPreamble != "" {
		// If errors occurred during bbolt check but preamble was found, return preamble with error
		if len(preambleConstructionErrors) > 0 {
			return cachedPreamble, fmt.Errorf("%w: %w", ErrAnalysisFailed, errors.Join(preambleConstructionErrors...))
		}
		return cachedPreamble, nil
	}
	opLogger.Debug("Preamble not found in bbolt cache or cache invalid, generating...")

	// --- Load base package and file info ---
	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)
	preambleConstructionErrors = append(preambleConstructionErrors, loadErrors...) // Add loading errors

	// Critical check: If package loading fails badly, we can't proceed.
	if targetPkg == nil {
		return "", fmt.Errorf("critical package load error for preamble: %w", errors.Join(preambleConstructionErrors...))
	}
	// targetFileAST and targetFile might be nil if the file itself has severe parsing issues
	// but the package context might still be somewhat loadable. Handle this gracefully.

	var cursorPos token.Pos = token.NoPos
	if targetFile != nil { // Only calculate cursor if targetFile is valid
		var posErr error
		cursorPos, posErr = calculateCursorPos(targetFile, line, col, opLogger)
		if posErr != nil {
			// This is a significant error for preamble context.
			preambleConstructionErrors = append(preambleConstructionErrors, fmt.Errorf("cannot calculate valid cursor position for preamble: %w", posErr))
			// Proceed with what we have, but preamble will be less accurate.
		}
	} else {
		preambleConstructionErrors = append(preambleConstructionErrors, errors.New("target token.File is nil, cursor-specific context will be limited"))
	}

	// --- Create AstContextInfo shell for this operation ---
	// This info struct will be populated by various Get* methods or their internal helpers.
	// It's also used by formatCursorContextSection.
	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		CursorPos:      cursorPos, // May be NoPos if targetFile was nil or calc failed
		TargetPackage:  targetPkg,
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST, // May be nil
		AnalysisErrors: make([]error, 0),
		Diagnostics:    loadDiagnostics[:],
		PackageName:    targetPkg.Name, // Populate package name
	}

	// --- Gather context using specific Get* methods ---
	enclosingCtx, encErr := a.GetEnclosingContext(ctx, absFilename, version, line, col)
	if encErr != nil {
		preambleConstructionErrors = append(preambleConstructionErrors, fmt.Errorf("enclosing context error: %w", encErr))
	}
	scopeInfo, scopeErr := a.GetScopeInfo(ctx, absFilename, version, line, col)
	if scopeErr != nil {
		preambleConstructionErrors = append(preambleConstructionErrors, fmt.Errorf("scope info error: %w", scopeErr))
	}
	comments, commentErr := a.GetRelevantComments(ctx, absFilename, version, line, col)
	if commentErr != nil {
		preambleConstructionErrors = append(preambleConstructionErrors, fmt.Errorf("comments error: %w", commentErr))
	}

	// --- Populate cursor-specific context nodes in tempInfo for formatCursorContextSection ---
	// This requires path finding and node identification.
	if tempInfo.TargetAstFile != nil && tempInfo.CursorPos.IsValid() {
		path, pathErr := findEnclosingPath(ctx, tempInfo.TargetAstFile, tempInfo.CursorPos, tempInfo, opLogger)
		if pathErr != nil {
			addAnalysisError(tempInfo, fmt.Errorf("finding path for cursor context failed: %w", pathErr), opLogger)
		}
		// findContextNodes populates tempInfo with CallExpr, SelectorExpr, CompositeLit, etc.
		findContextNodes(ctx, path, tempInfo.CursorPos, tempInfo.TargetPackage, tempInfo.TargetFileSet, a, tempInfo, opLogger)
		preambleConstructionErrors = append(preambleConstructionErrors, tempInfo.AnalysisErrors...) // Add errors from findContextNodes
	} else if tempInfo.TargetAstFile == nil {
		preambleConstructionErrors = append(preambleConstructionErrors, errors.New("target AST is nil, cannot determine specific cursor context nodes"))
	} else if !tempInfo.CursorPos.IsValid() {
		preambleConstructionErrors = append(preambleConstructionErrors, errors.New("cursor position invalid, cannot determine specific cursor context nodes"))
	}

	// --- Determine Qualifier ---
	var qualifier types.Qualifier
	if targetPkg != nil && targetPkg.Types != nil {
		qualifier = types.RelativeTo(targetPkg.Types)
	} else {
		// Fallback qualifier if package type info is unavailable
		qualifier = func(other *types.Package) string {
			if other != nil {
				return other.Path() // Or other.Name()
			}
			return ""
		}
	}

	// --- Build the Preamble String ---
	var preambleBuilder strings.Builder
	const internalPreambleLimit = 8192 // Max size for the constructed preamble parts
	currentLen := 0
	limitReached := false

	addToPreamble := func(s string) bool {
		if limitReached {
			return false
		}
		if currentLen+len(s) < internalPreambleLimit {
			preambleBuilder.WriteString(s)
			currentLen += len(s)
			return true
		}
		limitReached = true
		opLogger.Debug("Internal preamble construction limit reached", "limit", internalPreambleLimit)
		return false
	}
	addTruncMarker := func(section string) {
		if limitReached {
			return
		} // Only add marker if limit wasn't already hit
		msg := fmt.Sprintf("// ... (%s truncated)\n", section)
		// Check if adding marker itself exceeds limit
		if currentLen+len(msg) < internalPreambleLimit {
			preambleBuilder.WriteString(msg)
			currentLen += len(msg)
		}
		limitReached = true // Mark limit reached even if marker couldn't be added
	}

	// Add basic file/package info
	pkgName := tempInfo.PackageName
	if pkgName == "" {
		pkgName = "[unknown]"
	}
	addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(absFilename), pkgName))

	// Add Imports (still relies on tempInfo.TargetAstFile.Imports for now)
	// TODO: Potentially get imports from targetPkg.Imports if more robust
	if tempInfo.TargetAstFile != nil && !limitReached {
		tempInfo.Imports = tempInfo.TargetAstFile.Imports // Ensure tempInfo has imports if AST is available
		formatImportsSection(&preambleBuilder, tempInfo, addToPreamble, addTruncMarker, opLogger)
	}

	if !limitReached {
		formatEnclosingFuncSection(&preambleBuilder, enclosingCtx, qualifier, addToPreamble, opLogger)
	}
	if !limitReached {
		formatCommentsSection(&preambleBuilder, comments, addToPreamble, addTruncMarker, opLogger)
	}
	// Pass the tempInfo containing cursor context nodes (CallExpr, SelectorExpr etc.)
	if !limitReached {
		formatCursorContextSection(&preambleBuilder, tempInfo, qualifier, addToPreamble, opLogger)
	}
	if !limitReached {
		formatScopeSection(&preambleBuilder, scopeInfo, qualifier, addToPreamble, addTruncMarker, opLogger)
	}

	generatedPreamble := preambleBuilder.String()
	if generatedPreamble == "" {
		opLogger.Warn("Preamble generation resulted in empty string despite efforts.")
		// Potentially add a minimal fallback if everything failed.
		if len(preambleConstructionErrors) > 0 {
			generatedPreamble = "// Warning: Preamble generation encountered errors and resulted in empty output.\n"
		}
	}

	// --- Save to bbolt cache if generated successfully ---
	if generatedPreamble != "" && a.db != nil && len(loadErrors) == 0 { // Only save if no critical load errors
		saveStart := time.Now()
		// Recalculate inputHashes with targetPkg if available now
		inputHashes, hashErr := calculateInputHashes(dir, targetPkg, opLogger)
		if hashErr == nil {
			analysisDataToCache := CachedAnalysisData{PackageName: pkgName, PromptPreamble: generatedPreamble}
			var gobBuf bytes.Buffer
			if encodeErr := gob.NewEncoder(&gobBuf).Encode(&analysisDataToCache); encodeErr == nil {
				entryToSave := CachedAnalysisEntry{
					SchemaVersion:   cacheSchemaVersion,
					GoModHash:       goModHash,
					InputFileHashes: inputHashes,
					AnalysisGob:     gobBuf.Bytes(),
				}
				var entryBuf bytes.Buffer
				if entryEncodeErr := gob.NewEncoder(&entryBuf).Encode(&entryToSave); entryEncodeErr == nil {
					if saveErr := a.db.Update(func(tx *bbolt.Tx) error {
						b := tx.Bucket(cacheBucketName)
						if b == nil {
							return fmt.Errorf("bbolt bucket %s not found for saving preamble", cacheBucketName)
						}
						return b.Put(bboltCacheKey, entryBuf.Bytes())
					}); saveErr != nil {
						opLogger.Warn("Failed to save generated preamble to bbolt cache", "error", saveErr)
						preambleConstructionErrors = append(preambleConstructionErrors, fmt.Errorf("bbolt save error: %w", saveErr))
					} else {
						opLogger.Debug("Successfully saved generated preamble to bbolt cache", "duration", time.Since(saveStart))
					}
				} else {
					opLogger.Warn("Failed to GOB encode bbolt cache entry for preamble", "error", entryEncodeErr)
				}
			} else {
				opLogger.Warn("Failed to GOB encode preamble data for bbolt cache", "error", encodeErr)
			}
		} else {
			opLogger.Warn("Failed to calculate input file hashes for bbolt cache (preamble)", "error", hashErr)
		}
	}

	// Combine all non-fatal errors from construction.
	finalErr := errors.Join(preambleConstructionErrors...)
	if finalErr != nil {
		return generatedPreamble, fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr)
	}
	return generatedPreamble, nil
}

// GetMemoryCache implements the Analyzer interface method.
func (a *GoPackagesAnalyzer) GetMemoryCache(key string) (any, bool) {
	a.mu.RLock()
	cache := a.memoryCache
	a.mu.RUnlock()
	if cache == nil {
		a.logger.Debug("GetMemoryCache skipped: Cache is nil.", "key", key)
		return nil, false
	}
	val, found := cache.Get(key)
	if found {
		a.logger.Debug("GetMemoryCache hit.", "key", key)
	} else {
		a.logger.Debug("GetMemoryCache miss.", "key", key)
	}
	return val, found
}

// SetMemoryCache implements the Analyzer interface method.
func (a *GoPackagesAnalyzer) SetMemoryCache(key string, value any, cost int64, ttl time.Duration) bool {
	a.mu.RLock()
	cache := a.memoryCache
	a.mu.RUnlock()
	if cache == nil {
		a.logger.Debug("SetMemoryCache skipped: Cache is nil.", "key", key)
		return false
	}
	set := cache.SetWithTTL(key, value, cost, ttl)
	if set {
		a.logger.Debug("SetMemoryCache success.", "key", key, "cost", cost, "ttl", ttl)
	} else {
		// This can happen if item is too large vs. MaxCost or other Ristretto constraints.
		a.logger.Warn("SetMemoryCache failed (item might be too large or other constraint).", "key", key, "cost", cost, "ttl", ttl)
	}
	return set
}

// InvalidateCache removes the bbolt cached entry for a given directory.
// It specifically targets the cache key format used by GetPromptPreamble's bbolt caching.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	logger := a.logger.With("dir", dir, "op", "InvalidateCache")
	a.mu.RLock()
	db := a.db
	a.mu.RUnlock()

	if db == nil {
		logger.Debug("Bbolt cache invalidation skipped: DB is nil.")
		return nil
	}
	// Reconstruct the bbolt cache key as used in GetPromptPreamble
	goModHash := calculateGoModHash(dir, logger)
	bboltCacheKey := []byte(dir + "::" + goModHash)

	logger.Info("Invalidating bbolt cache entry", "key", string(bboltCacheKey))
	return deleteCacheEntryByKey(db, bboltCacheKey, logger)
}

// InvalidateMemoryCacheForURI clears relevant entries from the ristretto memory cache.
// Currently, it clears the entire memory cache due to complexity of tracking dependencies.
func (a *GoPackagesAnalyzer) InvalidateMemoryCacheForURI(uri string, version int) error {
	logger := a.logger.With("uri", uri, "version", version, "op", "InvalidateMemoryCacheForURI")
	a.mu.Lock() // Use write lock to safely access and clear cache
	memCache := a.memoryCache
	a.mu.Unlock()

	if memCache == nil {
		logger.Debug("Memory cache invalidation skipped: Cache is nil.")
		return nil
	}
	// TODO: Implement more granular invalidation if feasible.
	// For now, clearing the entire cache is a safe approach upon document change.
	logger.Warn("Clearing entire Ristretto memory cache due to document change.", "uri", uri)
	memCache.Clear() // Clears all items from the cache
	return nil
}

// MemoryCacheEnabled returns true if the Ristretto cache is initialized and available.
func (a *GoPackagesAnalyzer) MemoryCacheEnabled() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.memoryCache != nil
}

// GetMemoryCacheMetrics returns the performance metrics collected by Ristretto.
func (a *GoPackagesAnalyzer) GetMemoryCacheMetrics() *ristretto.Metrics {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.memoryCache != nil {
		return a.memoryCache.Metrics
	}
	return nil // Return nil if cache is not enabled
}

// --- Default Prompt Formatter ---

// templateFormatter implements the PromptFormatter interface using standard Go templates.
type templateFormatter struct{}

// newTemplateFormatter creates a new instance of the default formatter.
func newTemplateFormatter() *templateFormatter { return &templateFormatter{} }

// FormatPrompt combines the context preamble and code snippet into the final LLM prompt string.
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config, logger *stdslog.Logger) string {
	if logger == nil {
		logger = stdslog.Default()
	}
	var finalPreamble strings.Builder
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2 // Split snippet length for FIM prefix/suffix

	// --- Add Fallback Context if present (e.g., from non-AST analysis) ---
	if snippetCtx.FallbackContext != "" {
		truncatedFallback := snippetCtx.FallbackContext
		// Truncate fallback context if it exceeds maxPreambleLen
		if len(truncatedFallback) > maxPreambleLen {
			logger.Warn("Truncating fallback context", "original_length", len(truncatedFallback), "max_length", maxPreambleLen)
			marker := "// ... (Fallback context truncated)\n"
			// Truncate from the beginning to keep the most recent part
			startByte := len(truncatedFallback) - maxPreambleLen + len(marker)
			if startByte < 0 {
				startByte = 0 // Ensure startByte is not negative
			}
			// Ensure marker fits
			if len(marker)+len(truncatedFallback[startByte:]) > maxPreambleLen {
				// If marker + remaining doesn't fit, just take tail of original
				truncatedFallback = truncatedFallback[len(truncatedFallback)-maxPreambleLen:]
			} else {
				truncatedFallback = marker + truncatedFallback[startByte:]
			}
		}
		finalPreamble.WriteString(truncatedFallback)
		finalPreamble.WriteString("\n---\n") // Separator between fallback and main preamble
	}

	// --- Add Main Preamble (AST-based or default) ---
	truncatedPreamble := contextPreamble
	if len(truncatedPreamble) > maxPreambleLen {
		logger.Warn("Truncating context preamble", "original_length", len(truncatedPreamble), "max_length", maxPreambleLen)
		marker := "... (context truncated)\n" // Truncation marker
		// Truncate from the beginning to keep the most recent part of the preamble
		startByte := len(truncatedPreamble) - maxPreambleLen + len(marker)
		if startByte < 0 {
			startByte = 0
		}
		if len(marker)+len(truncatedPreamble[startByte:]) > maxPreambleLen {
			truncatedPreamble = truncatedPreamble[len(truncatedPreamble)-maxPreambleLen:]
		} else {
			truncatedPreamble = marker + truncatedPreamble[startByte:]
		}
	}
	finalPreamble.WriteString(truncatedPreamble)

	// --- Format Snippet based on FIM ---
	prefix := snippetCtx.Prefix
	suffix := snippetCtx.Suffix

	if config.UseFim {
		template := config.FimTemplate
		// Truncate FIM prefix from the beginning
		if len(prefix) > maxFIMPartLen {
			logger.Warn("Truncating FIM prefix", "original_length", len(prefix), "max_length", maxFIMPartLen)
			marker := "...(prefix truncated)"
			startByte := len(prefix) - maxFIMPartLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			if len(marker)+len(prefix[startByte:]) > maxFIMPartLen {
				prefix = prefix[len(prefix)-maxFIMPartLen:]
			} else {
				prefix = marker + prefix[startByte:]
			}
		}
		// Truncate FIM suffix from the end
		if len(suffix) > maxFIMPartLen {
			logger.Warn("Truncating FIM suffix", "original_length", len(suffix), "max_length", maxFIMPartLen)
			marker := "(suffix truncated)..."
			endByte := maxFIMPartLen - len(marker)
			if endByte < 0 {
				endByte = 0
			}
			if len(suffix[:endByte])+len(marker) > maxFIMPartLen {
				suffix = suffix[:maxFIMPartLen]
			} else {
				suffix = suffix[:endByte] + marker
			}
		}
		return fmt.Sprintf(template, finalPreamble.String(), prefix, suffix)
	} else { // Standard completion
		template := config.PromptTemplate
		snippet := prefix // For standard completion, only prefix is used as the snippet
		if len(snippet) > maxSnippetLen {
			logger.Warn("Truncating code snippet (prefix)", "original_length", len(snippet), "max_length", maxSnippetLen)
			marker := "...(code truncated)\n"
			// Truncate from the beginning
			startByte := len(snippet) - maxSnippetLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			if len(marker)+len(snippet[startByte:]) > maxSnippetLen {
				snippet = snippet[len(snippet)-maxSnippetLen:]
			} else {
				snippet = marker + snippet[startByte:]
			}
		}
		return fmt.Sprintf(template, finalPreamble.String(), snippet)
	}
}

// =============================================================================
// DeepCompleter Service
// =============================================================================

// DeepCompleter orchestrates the code analysis, prompt formatting, and LLM interaction.
type DeepCompleter struct {
	client    LLMClient       // Interface for interacting with the LLM backend.
	analyzer  Analyzer        // Interface for performing code analysis.
	formatter PromptFormatter // Interface for formatting the LLM prompt.
	config    Config          // Current active configuration
	configMu  sync.RWMutex    // Mutex to protect concurrent access to config.
	logger    *stdslog.Logger // Logger instance for the DeepCompleter service
}

// NewDeepCompleter creates a new DeepCompleter service instance.
// It loads configuration and initializes components.
func NewDeepCompleter(logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
	serviceLogger := logger.With("service", "DeepCompleter")

	// Load initial configuration
	cfg, configErr := LoadConfig(serviceLogger) // configErr can be ErrConfig (non-fatal) or other fatal errors
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		// Fatal error during initial config load (e.g., cannot determine paths, default config invalid)
		serviceLogger.Error("Fatal error during initial config load", "error", configErr)
		return nil, configErr
	}
	// Validate the loaded/defaulted config
	if err := cfg.Validate(serviceLogger); err != nil {
		serviceLogger.Error("Initial configuration is invalid after loading/defaults", "error", err)
		// If validation itself returns ErrInvalidConfig, it's a structured validation failure.
		// Otherwise, it might be an unexpected issue during validation.
		if errors.Is(err, ErrInvalidConfig) {
			return nil, fmt.Errorf("initial config validation failed: %w", err)
		}
		// Log other validation issues as warnings, but proceed if possible (Validate might fix some)
		serviceLogger.Warn("Initial config validation reported issues", "error", err)
		// If configErr was already ErrConfig, combine them.
		if configErr != nil {
			configErr = errors.Join(configErr, err)
		} else {
			configErr = err // Assign the validation error
		}
	}

	// Initialize analyzer with the (potentially fixed by Validate) config
	analyzer := NewGoPackagesAnalyzer(serviceLogger, cfg)
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    cfg, // Store the initial, validated config
		logger:    serviceLogger,
	}

	// Return the DeepCompleter instance and any non-fatal ErrConfig from LoadConfig/Validate
	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil // No fatal error, and no ErrConfig, or ErrConfig was handled.
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with a specific config.
// This is useful for testing or specific programmatic setups.
func NewDeepCompleterWithConfig(config Config, logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
	serviceLogger := logger.With("service", "DeepCompleter")

	// Ensure default templates are set if not provided
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	// Validate the provided config
	if err := config.Validate(serviceLogger); err != nil {
		return nil, fmt.Errorf("provided config validation failed: %w", err)
	}

	analyzer := NewGoPackagesAnalyzer(serviceLogger, config) // Pass validated config to analyzer
	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    config, // Store the validated config
		logger:    serviceLogger,
	}, nil
}

// Close cleans up resources used by the DeepCompleter, primarily the analyzer.
func (dc *DeepCompleter) Close() error {
	dc.logger.Info("Closing DeepCompleter service")
	if dc.analyzer != nil {
		return dc.analyzer.Close() // Delegate closing to the analyzer
	}
	return nil
}

// UpdateConfig atomically updates the completer's configuration.
// It validates the new configuration before applying it.
func (dc *DeepCompleter) UpdateConfig(newConfig Config) error {
	// Ensure default templates are set if not provided in the update
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate
	}
	// Validate the new configuration
	if err := newConfig.Validate(dc.logger); err != nil {
		dc.logger.Error("Invalid configuration provided for update", "error", err)
		return fmt.Errorf("invalid configuration update: %w", err) // Return validation error
	}

	dc.configMu.Lock()
	dc.config = newConfig // Apply the new, validated config
	dc.configMu.Unlock()

	// Propagate config update to the analyzer
	if dc.analyzer != nil {
		dc.analyzer.UpdateConfig(newConfig)
	}

	dc.logger.Info("DeepCompleter configuration updated",
		stdslog.Group("new_config", // Log relevant parts of the new config
			stdslog.String("ollama_url", newConfig.OllamaURL),
			stdslog.String("model", newConfig.Model),
			stdslog.Int("max_tokens", newConfig.MaxTokens),
			stdslog.Any("stop", newConfig.Stop), // slog.Any for slices
			stdslog.Float64("temperature", newConfig.Temperature),
			stdslog.String("log_level", newConfig.LogLevel),
			stdslog.Bool("use_ast", newConfig.UseAst),
			stdslog.Bool("use_fim", newConfig.UseFim),
			stdslog.Int("max_preamble_len", newConfig.MaxPreambleLen),
			stdslog.Int("max_snippet_len", newConfig.MaxSnippetLen),
			stdslog.Int("memory_cache_ttl_seconds", newConfig.MemoryCacheTTLSeconds),
		),
	)
	return nil
}

// GetCurrentConfig returns a thread-safe copy of the current configuration.
// This prevents direct modification of the internal config state.
func (dc *DeepCompleter) GetCurrentConfig() Config {
	dc.configMu.RLock()
	defer dc.configMu.RUnlock()
	cfgCopy := dc.config // Create a copy
	// Ensure slice fields are also copied to prevent modification by reference
	if cfgCopy.Stop != nil {
		stopsCopy := make([]string, len(cfgCopy.Stop))
		copy(stopsCopy, cfgCopy.Stop)
		cfgCopy.Stop = stopsCopy
	}
	return cfgCopy
}

// Client returns the LLMClient instance used by the completer.
func (dc *DeepCompleter) Client() LLMClient {
	return dc.client
}

// InvalidateAnalyzerCache provides external access to invalidate the analyzer's disk cache for a directory.
func (dc *DeepCompleter) InvalidateAnalyzerCache(dir string) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized, cannot invalidate cache")
	}
	dc.logger.Info("Request to invalidate analyzer disk cache", "dir", dir)
	return dc.analyzer.InvalidateCache(dir)
}

// InvalidateMemoryCacheForURI provides external access to invalidate the analyzer's memory cache for a URI.
func (dc *DeepCompleter) InvalidateMemoryCacheForURI(uri string, version int) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized, cannot invalidate memory cache")
	}
	dc.logger.Info("Request to invalidate analyzer memory cache", "uri", uri, "version", version)
	return dc.analyzer.InvalidateMemoryCacheForURI(uri, version)
}

// GetAnalyzer returns the analyzer instance, useful for direct interaction (e.g., testing).
func (dc *DeepCompleter) GetAnalyzer() Analyzer {
	return dc.analyzer
}

// GetCompletion provides basic code completion for a given snippet without file context analysis.
// This is a simpler completion path, typically for quick, non-contextual suggestions.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	opLogger := dc.logger.With("operation", "GetCompletion")
	opLogger.Info("Handling basic completion request")
	currentConfig := dc.GetCurrentConfig() // Get a thread-safe copy of config

	// Basic preamble for non-AST based completion
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet, Suffix: "", FullLine: ""}

	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig, opLogger)
	opLogger.Debug("Generated basic prompt", "length", len(prompt))

	var buffer bytes.Buffer // Buffer to store completion result
	// apiCallFunc defines the operation to be retried
	apiCallFunc := func() error {
		select {
		case <-ctx.Done(): // Check for context cancellation before API call
			return ctx.Err()
		default:
		}
		// Use a timeout for the API call itself, derived from the outer context
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()

		opLogger.Debug("Calling Ollama GenerateStream for basic completion")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig, opLogger)
		if apiErr != nil {
			return apiErr // Return error to be handled by retry logic
		}
		defer reader.Close()

		// Use a timeout for processing the stream
		streamCtx, cancelStream := context.WithTimeout(apiCtx, 50*time.Second)
		defer cancelStream()

		buffer.Reset() // Reset buffer for each attempt
		streamErr := streamCompletion(streamCtx, reader, &buffer, opLogger)
		if streamErr != nil {
			return fmt.Errorf("streaming completion failed: %w", streamErr)
		}
		return nil
	}

	// Retry the API call operation
	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, opLogger)
	if err != nil {
		select {
		case <-ctx.Done(): // Check if overall context was cancelled
			return "", ctx.Err()
		default:
		}
		// Handle specific error types after retries
		if errors.Is(err, ErrOllamaUnavailable) {
			opLogger.Error("Ollama unavailable for basic completion after retries", "error", err)
			return "", err
		}
		if errors.Is(err, ErrStreamProcessing) {
			opLogger.Error("Stream processing error for basic completion after retries", "error", err)
			return "", err
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			opLogger.Warn("Basic completion context cancelled or timed out after retries", "error", err)
			return "", err
		}
		opLogger.Error("Failed to get basic completion after retries", "error", err)
		return "", fmt.Errorf("failed to get basic completion after %d retries: %w", maxRetries, err)
	}

	opLogger.Info("Basic completion successful")
	return strings.TrimSpace(buffer.String()), nil
}

// GetCompletionStreamFromFile provides context-aware code completion by analyzing the
// specified file and position, then streams the LLM response to the provided writer.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, absFilename string, version int, line, col int, w io.Writer) error {
	opLogger := dc.logger.With("operation", "GetCompletionStreamFromFile", "path", absFilename, "version", version, "line", line, "col", col)
	currentConfig := dc.GetCurrentConfig()
	var contextPreamble string = "// Basic file context only." // Default preamble if AST analysis fails or is disabled
	var preambleErr error                                      // To store non-fatal errors from preamble generation

	if currentConfig.UseAst {
		opLogger.Info("Analyzing context for preamble (or checking cache)")
		preambleCtx, cancelPreamble := context.WithTimeout(ctx, 30*time.Second) // Timeout for preamble generation
		contextPreamble, preambleErr = dc.analyzer.GetPromptPreamble(preambleCtx, absFilename, version, line, col)
		cancelPreamble() // Ensure cancel is called

		if preambleErr != nil && !errors.Is(preambleErr, ErrAnalysisFailed) {
			// Fatal error during preamble generation
			opLogger.Error("Fatal error getting prompt preamble", "error", preambleErr)
			return fmt.Errorf("failed to get prompt preamble: %w", preambleErr)
		}
		if preambleErr != nil { // Non-fatal ErrAnalysisFailed
			opLogger.Warn("Non-fatal error getting prompt preamble", "error", preambleErr)
			// Prepend warning to the potentially partial preamble
			contextPreamble += fmt.Sprintf("\n// Warning: Preamble generation completed with errors: %v\n", preambleErr)
		}
		if contextPreamble == "" { // Fallback if preamble is empty after analysis
			contextPreamble = "// Warning: Context analysis returned empty preamble.\n"
			opLogger.Warn("Context analysis resulted in an empty preamble string.")
		}
	} else {
		opLogger.Info("AST analysis disabled by config. Using basic preamble.")
	}

	// Extract code snippet context (prefix, suffix, etc.)
	// Pass currentConfig.UseAst to extractSnippetContext so it knows whether to attempt fallback context
	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col, currentConfig.UseAst, opLogger)
	if snippetErr != nil {
		opLogger.Error("Failed to extract code snippet context", "error", snippetErr)
		return fmt.Errorf("failed to extract code snippet context: %w", snippetErr)
	}

	// Format the final prompt using the gathered preamble and snippet context
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig, opLogger)
	opLogger.Debug("Generated prompt", "length", len(prompt))

	// Define the API call operation for retry logic
	apiCallFunc := func() error {
		select {
		case <-ctx.Done(): // Check for context cancellation
			return ctx.Err()
		default:
		}
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second) // Timeout for the LLM call
		defer cancelApi()
		opLogger.Debug("Calling Ollama GenerateStream")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig, opLogger)
		if apiErr != nil {
			return apiErr // Error to be handled by retry
		}
		defer reader.Close()
		// Stream the completion to the provided writer
		streamErr := streamCompletion(apiCtx, reader, w, opLogger) // Pass apiCtx for stream processing timeout
		if streamErr != nil {
			return fmt.Errorf("streaming completion failed: %w", streamErr)
		}
		return nil
	}

	// Retry the API call
	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, opLogger)
	if err != nil {
		select {
		case <-ctx.Done(): // Check overall context cancellation
			return ctx.Err()
		default:
		}
		// Handle specific errors after retries
		if errors.Is(err, ErrOllamaUnavailable) {
			opLogger.Error("Ollama unavailable for stream after retries", "error", err)
			return err
		}
		if errors.Is(err, ErrStreamProcessing) {
			opLogger.Error("Stream processing error for stream after retries", "error", err)
			return err
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			opLogger.Warn("Completion stream context cancelled or timed out after retries", "error", err)
			return err
		}
		opLogger.Error("Failed to get completion stream after retries", "error", err)
		return fmt.Errorf("failed to get completion stream after %d retries: %w", maxRetries, err)
	}

	// Log preamble errors even if completion succeeded, as they indicate partial context
	if preambleErr != nil {
		opLogger.Warn("Completion stream successful, but context analysis for preamble encountered non-fatal errors", "analysis_error", preambleErr)
	} else {
		opLogger.Info("Completion stream successful")
	}
	return nil
}
