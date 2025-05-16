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
					loadDuration = time.Since(decodeStart)
					analysisLogger.Debug("Preamble successfully decoded from bbolt cache.", "duration", loadDuration)
					cacheHit = false // For Analyze, a bbolt hit for preamble doesn't mean full analysis is cached.
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
		return info, analysisErr // Return info and allow defer to set analysisErr
	}
	info.TargetPackage = targetPkg
	info.TargetAstFile = targetFileAST
	// --- End Load Package Info ---

	// --- Perform Analysis Steps (using loaded info) ---
	stepsStart := time.Now()
	if targetFile != nil { // Ensure targetFile (token.File) is valid
		analyzeStepErr := performAnalysisSteps(ctx, targetFile, targetFileAST, targetPkg, fset, line, col, a, info, analysisLogger)
		if analyzeStepErr != nil {
			analysisLogger.Error("Fatal error during performAnalysisSteps", "error", analyzeStepErr)
			analysisErr = analyzeStepErr // Set the fatal error
			return info, analysisErr     // Return immediately
		}
	} else {
		if len(loadErrors) == 0 {
			addAnalysisError(info, errors.New("cannot perform analysis steps: target token.File is nil"), analysisLogger)
		}
	}
	stepsDuration = time.Since(stepsStart)
	analysisLogger.Debug("Analysis steps completed", "duration", stepsDuration)
	// --- End Analysis Steps ---

	// --- Construct Preamble (if not loaded from bbolt cache) ---
	if info.PromptPreamble == "" {
		preambleStart := time.Now()
		preamble, preambleErr := a.GetPromptPreamble(ctx, absFilename, version, line, col)
		if preambleErr != nil {
			analysisLogger.Error("Error getting prompt preamble", "error", preambleErr)
			addAnalysisError(info, fmt.Errorf("failed to get prompt preamble: %w", preambleErr), analysisLogger)
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
	shouldSave := a.db != nil && info.PromptPreamble != "" && !cacheHit && len(loadErrors) == 0
	if shouldSave {
		analysisLogger.Debug("Attempting to save analysis results to bbolt cache.", "key", string(cacheKey))
		saveStart := time.Now()
		inputHashes, hashErr := calculateInputHashes(dir, targetPkg, analysisLogger)
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

	return info, analysisErr
}

// GetIdentifierInfo implements the Analyzer interface method.
// It attempts to find detailed information about the identifier at the given cursor position.
func (a *GoPackagesAnalyzer) GetIdentifierInfo(ctx context.Context, absFilename string, version int, line, col int) (*IdentifierInfo, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetIdentifierInfo")
	opLogger.Info("Starting identifier analysis")

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
		combinedErr := errors.Join(loadErrors...)
		opLogger.Error("Critical error loading package, cannot get identifier info", "error", combinedErr)
		return nil, fmt.Errorf("critical package load error: %w", combinedErr)
	}
	if targetFile == nil || targetFileAST == nil {
		errText := "target file AST or token.File is nil after loading"
		opLogger.Error(errText, "targetFile_nil", targetFile == nil, "targetAST_nil", targetFileAST == nil)
		return nil, errors.Join(append(loadErrors, errors.New(errText))...)
	}
	if targetPkg.TypesInfo == nil {
		opLogger.Warn("Type info (TypesInfo) is nil for the package. Identifier resolution will likely fail or be incomplete.")
		addAnalysisError(tempInfo, errors.New("package type information (TypesInfo) is missing"), opLogger)
	}

	cursorPos, posErr := calculateCursorPos(targetFile, line, col, opLogger)
	if posErr != nil {
		return nil, fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		return nil, fmt.Errorf("%w: invalid cursor position calculated (Pos: %d)", ErrPositionConversion, cursorPos)
	}
	tempInfo.CursorPos = cursorPos
	opLogger = opLogger.With("cursorPos", cursorPos, "cursorPosStr", fset.PositionFor(cursorPos, true).String())

	path, pathErr := findEnclosingPath(ctx, targetFileAST, cursorPos, tempInfo, opLogger)
	if pathErr != nil {
		opLogger.Warn("Error finding enclosing AST path", "error", pathErr)
		addAnalysisError(tempInfo, fmt.Errorf("finding enclosing AST path failed: %w", pathErr), opLogger)
	}
	findContextNodes(ctx, path, cursorPos, targetPkg, fset, a, tempInfo, opLogger)

	analysisProcessError := errors.Join(tempInfo.AnalysisErrors...)
	if analysisProcessError != nil {
		opLogger.Warn("Non-fatal errors occurred during identifier analysis steps", "errors", analysisProcessError)
		analysisProcessError = fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisProcessError)
	}

	if tempInfo.IdentifierAtCursor == nil || tempInfo.IdentifierObject == nil {
		opLogger.Debug("No specific identifier object found at cursor position.")
		return nil, analysisProcessError
	}

	fileContent, readErr := os.ReadFile(absFilename)
	if readErr != nil {
		opLogger.Error("Failed to read file content for identifier info construction", "error", readErr)
		finalErr := errors.Join(analysisProcessError, fmt.Errorf("failed to read file content %s: %w", absFilename, readErr))
		return nil, finalErr
	}

	identInfo := &IdentifierInfo{
		Name:      tempInfo.IdentifierObject.Name(),
		Object:    tempInfo.IdentifierObject,
		Type:      tempInfo.IdentifierType,
		DefNode:   tempInfo.IdentifierDefNode,
		FileSet:   fset,
		Pkg:       targetPkg,
		Content:   fileContent,
		IdentNode: tempInfo.IdentifierAtCursor,
	}

	if defPos := identInfo.Object.Pos(); defPos.IsValid() {
		identInfo.DefFilePos = fset.Position(defPos)
	}

	opLogger.Info("Identifier info retrieved successfully", "identifier", identInfo.Name)
	return identInfo, analysisProcessError
}

// GetEnclosingContext implements the Analyzer interface method.
// It retrieves information about the function, method, or block enclosing the cursor.
func (a *GoPackagesAnalyzer) GetEnclosingContext(ctx context.Context, absFilename string, version int, line, col int) (*EnclosingContextInfo, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetEnclosingContext")
	opLogger.Debug("Starting enclosing context analysis")

	// 1. Load necessary package and AST data
	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)

	// Initialize a temporary AstContextInfo to hold data for this operation
	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		TargetPackage:  targetPkg,
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST,
		AnalysisErrors: make([]error, 0),   // Collect non-fatal errors here
		Diagnostics:    loadDiagnostics[:], // Start with diagnostics from loading
	}

	// Handle critical loading errors
	if targetPkg == nil {
		return nil, fmt.Errorf("critical package load error: %w", errors.Join(loadErrors...))
	}
	if targetFile == nil || targetFileAST == nil {
		return nil, errors.Join(append(loadErrors, errors.New("target file AST or token.File is nil"))...)
	}

	// 2. Calculate cursor position
	cursorPos, posErr := calculateCursorPos(targetFile, line, col, opLogger)
	if posErr != nil {
		return nil, fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
	}
	tempInfo.CursorPos = cursorPos // Store valid cursor position

	// 3. Find AST path and gather context using a cached computation
	// The computePathFn will find the AST path.
	// gatherScopeContext (called after path finding) populates EnclosingFunc, EnclosingFuncNode, etc., in tempInfo.
	cacheKey := generateCacheKey("enclosingContext", tempInfo) // Use a specific key for this operation
	computeEnclosingContextFn := func() (*EnclosingContextInfo, error) {
		// This function will be wrapped by withMemoryCache.
		// It needs to perform path finding and then call gatherScopeContext.

		// Create a computation-local AstContextInfo to avoid side effects if outer tempInfo is used elsewhere.
		// Or, ensure that tempInfo passed to gatherScopeContext is correctly scoped.
		// For this specific getter, we are interested in fields populated by gatherScopeContext.
		localTempInfo := &AstContextInfo{ // Create a fresh one for the computation if needed, or pass/reset parts of outer tempInfo
			FilePath:       tempInfo.FilePath,
			Version:        tempInfo.Version,
			CursorPos:      tempInfo.CursorPos,
			TargetPackage:  tempInfo.TargetPackage,
			TargetFileSet:  tempInfo.TargetFileSet,
			TargetAstFile:  tempInfo.TargetAstFile,
			AnalysisErrors: make([]error, 0),
		}

		path, pathErr := findEnclosingPath(ctx, localTempInfo.TargetAstFile, localTempInfo.CursorPos, localTempInfo, opLogger)
		if pathErr != nil {
			// Add to localTempInfo's errors, which will be part of the returned error from computeFn
			addAnalysisError(localTempInfo, fmt.Errorf("finding enclosing AST path failed: %w", pathErr), opLogger)
		}

		// gatherScopeContext populates EnclosingFunc, EnclosingFuncNode, ReceiverType, EnclosingBlock
		// on the localTempInfo struct.
		gatherScopeContext(ctx, path, localTempInfo.TargetPackage, localTempInfo.TargetFileSet, localTempInfo, opLogger)

		// Construct the EnclosingContextInfo from the populated localTempInfo
		result := &EnclosingContextInfo{
			Func:     localTempInfo.EnclosingFunc,
			FuncNode: localTempInfo.EnclosingFuncNode,
			Receiver: localTempInfo.ReceiverType,
			Block:    localTempInfo.EnclosingBlock,
		}

		// Combine any errors collected during this computation
		computationErr := errors.Join(localTempInfo.AnalysisErrors...)
		if computationErr != nil {
			return result, fmt.Errorf("%w: %w", ErrAnalysisFailed, computationErr)
		}
		return result, nil
	}

	// Use withMemoryCache to get/compute the EnclosingContextInfo
	enclosingInfo, cacheHit, combinedErr := withMemoryCache[*EnclosingContextInfo](
		a, cacheKey, 5, a.getConfig().MemoryCacheTTL, computeEnclosingContextFn, opLogger,
	)

	if cacheHit {
		opLogger.Debug("Enclosing context cache hit")
	}
	if combinedErr != nil {
		opLogger.Warn("Non-fatal errors during enclosing context analysis", "error", combinedErr)
		// Return the (potentially partially filled) enclosingInfo along with the error
		return enclosingInfo, combinedErr
	}

	return enclosingInfo, nil
}

// GetScopeInfo implements the Analyzer interface method.
// It retrieves variables, constants, types, and functions available in the scope at the cursor.
func (a *GoPackagesAnalyzer) GetScopeInfo(ctx context.Context, absFilename string, version int, line, col int) (*ScopeInfo, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetScopeInfo")
	opLogger.Debug("Starting scope analysis")

	// 1. Load necessary package and AST data
	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)

	// Initialize a temporary AstContextInfo for this operation
	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		TargetPackage:  targetPkg,
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST,
		AnalysisErrors: make([]error, 0),
		Diagnostics:    loadDiagnostics[:],
	}

	// Handle critical loading errors
	if targetPkg == nil {
		return nil, fmt.Errorf("critical package load error: %w", errors.Join(loadErrors...))
	}
	// targetFile can be nil if the file has severe parsing issues but package context is somewhat available.
	// Scope info might still be partially obtainable from package scope.
	if targetFile == nil {
		opLogger.Warn("Target token.File is nil; scope info will be limited to package scope if AST-based path finding fails.")
		// Add a non-fatal error to indicate potential incompleteness
		addAnalysisError(tempInfo, errors.New("target token.File is nil, limiting scope analysis"), opLogger)
	}

	// 2. Calculate cursor position (only if targetFile is valid)
	var cursorPos token.Pos = token.NoPos
	if targetFile != nil {
		var posErr error
		cursorPos, posErr = calculateCursorPos(targetFile, line, col, opLogger)
		if posErr != nil {
			// If cursor calculation fails, we might still get package scope, but not local scopes.
			addAnalysisError(tempInfo, fmt.Errorf("cannot calculate valid cursor position, local scope analysis will be affected: %w", posErr), opLogger)
			// Proceed with NoPos, gatherScopeContext should handle it by primarily giving package scope.
		}
	}
	tempInfo.CursorPos = cursorPos // Store valid or NoPos cursor position

	// 3. Use memory cache for scope extraction.
	// The computeScopeFn will find the AST path (if AST and cursor are valid) and then gather scope.
	cacheKey := generateCacheKey("scopeInfo", tempInfo) // Key includes cursor pos, so NoPos will have a different key
	computeScopeFn := func() (map[string]types.Object, error) {
		// Create a computation-local AstContextInfo
		computationTempInfo := &AstContextInfo{
			FilePath:         tempInfo.FilePath,
			Version:          tempInfo.Version,
			CursorPos:        tempInfo.CursorPos, // Use the (potentially NoPos) cursor
			TargetPackage:    tempInfo.TargetPackage,
			TargetFileSet:    tempInfo.TargetFileSet,
			TargetAstFile:    tempInfo.TargetAstFile,        // Can be nil
			VariablesInScope: make(map[string]types.Object), // Fresh map for this computation
			AnalysisErrors:   make([]error, 0),
		}

		var path []ast.Node
		var pathErr error
		if computationTempInfo.TargetAstFile != nil && computationTempInfo.CursorPos.IsValid() {
			path, pathErr = findEnclosingPath(ctx, computationTempInfo.TargetAstFile, computationTempInfo.CursorPos, computationTempInfo, opLogger)
			if pathErr != nil {
				addAnalysisError(computationTempInfo, fmt.Errorf("finding enclosing path failed for scope: %w", pathErr), opLogger)
			}
		} else if computationTempInfo.TargetAstFile == nil {
			addAnalysisError(computationTempInfo, errors.New("target AST is nil, cannot find path for local scope analysis"), opLogger)
		} else { // CursorPos is invalid
			addAnalysisError(computationTempInfo, errors.New("cursor position invalid, cannot find path for local scope analysis"), opLogger)
		}

		// gatherScopeContext populates computationTempInfo.VariablesInScope
		// It handles nil path or invalid cursorPos by primarily adding package scope.
		gatherScopeContext(ctx, path, computationTempInfo.TargetPackage, computationTempInfo.TargetFileSet, computationTempInfo, opLogger)

		computationErr := errors.Join(computationTempInfo.AnalysisErrors...)
		if computationErr != nil {
			// Return the map (even if partially filled) and the error
			return computationTempInfo.VariablesInScope, fmt.Errorf("errors during scope computation: %w", computationErr)
		}
		return computationTempInfo.VariablesInScope, nil
	}

	scopeMap, cacheHit, combinedErr := withMemoryCache[map[string]types.Object](
		a, cacheKey, 10, a.getConfig().MemoryCacheTTL, computeScopeFn, opLogger,
	)

	if cacheHit {
		opLogger.Debug("Scope info cache hit")
	}

	// Combine errors from loading/cursor calculation (in tempInfo) with errors from computation (in combinedErr)
	finalAnalysisErrors := tempInfo.AnalysisErrors
	if combinedErr != nil {
		finalAnalysisErrors = append(finalAnalysisErrors, combinedErr)
		opLogger.Warn("Non-fatal errors during scope extraction", "error", combinedErr)
	}

	finalCombinedErr := errors.Join(finalAnalysisErrors...)
	if finalCombinedErr != nil {
		return &ScopeInfo{Variables: scopeMap}, fmt.Errorf("%w: %w", ErrAnalysisFailed, finalCombinedErr)
	}
	return &ScopeInfo{Variables: scopeMap}, nil
}

// GetRelevantComments implements the Analyzer interface method.
func (a *GoPackagesAnalyzer) GetRelevantComments(ctx context.Context, absFilename string, version int, line, col int) ([]string, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetRelevantComments")
	opLogger.Debug("Starting comment analysis")

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

	if targetFileAST == nil || targetFile == nil {
		return nil, errors.Join(append(loadErrors, errors.New("target file AST or token.File is nil, cannot get comments"))...)
	}

	cursorPos, posErr := calculateCursorPos(targetFile, line, col, opLogger)
	if posErr != nil {
		return nil, fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
	}
	tempInfo.CursorPos = cursorPos

	cacheKey := generateCacheKey("comments", tempInfo)
	computeCommentsFn := func() ([]string, error) {
		computationTempInfo := &AstContextInfo{
			FilePath:           tempInfo.FilePath,
			Version:            tempInfo.Version,
			CursorPos:          tempInfo.CursorPos,
			TargetPackage:      tempInfo.TargetPackage,
			TargetFileSet:      tempInfo.TargetFileSet,
			TargetAstFile:      tempInfo.TargetAstFile,
			CommentsNearCursor: nil,
			AnalysisErrors:     make([]error, 0),
		}

		path, pathErr := findEnclosingPath(ctx, computationTempInfo.TargetAstFile, computationTempInfo.CursorPos, computationTempInfo, opLogger)
		if pathErr != nil {
			addAnalysisError(computationTempInfo, fmt.Errorf("finding enclosing path failed for comments: %w", pathErr), opLogger)
		}

		findRelevantComments(ctx, computationTempInfo.TargetAstFile, path, computationTempInfo.CursorPos, computationTempInfo.TargetFileSet, computationTempInfo, opLogger)

		computationErr := errors.Join(computationTempInfo.AnalysisErrors...)
		if computationErr != nil {
			return computationTempInfo.CommentsNearCursor, fmt.Errorf("errors during comment computation: %w", computationErr)
		}
		return computationTempInfo.CommentsNearCursor, nil
	}

	comments, cacheHit, commentErr := withMemoryCache[[]string](a, cacheKey, 5, a.getConfig().MemoryCacheTTL, computeCommentsFn, opLogger)

	finalAnalysisErrors := tempInfo.AnalysisErrors // Start with errors from loading/cursor calc
	if commentErr != nil {
		finalAnalysisErrors = append(finalAnalysisErrors, commentErr)
		opLogger.Warn("Non-fatal errors during comment extraction", "error", commentErr)
	}
	if cacheHit {
		opLogger.Debug("Relevant comments cache hit")
	}

	finalCombinedErr := errors.Join(finalAnalysisErrors...)
	if finalCombinedErr != nil {
		return comments, fmt.Errorf("%w: %w", ErrAnalysisFailed, finalCombinedErr)
	}
	return comments, nil
}

// GetPromptPreamble implements the Analyzer interface method.
// It orchestrates calls to other Get* methods to gather context and then formats it.
func (a *GoPackagesAnalyzer) GetPromptPreamble(ctx context.Context, absFilename string, version int, line, col int) (string, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetPromptPreamble")
	opLogger.Debug("Starting preamble generation")

	dir := filepath.Dir(absFilename)
	goModHash := calculateGoModHash(dir, opLogger)
	bboltCacheKey := []byte(dir + "::" + goModHash)
	var cachedPreamble string
	var preambleConstructionErrors []error

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
			currentHashes, hashErr := calculateInputHashes(dir, nil, opLogger)
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
		if len(preambleConstructionErrors) > 0 {
			return cachedPreamble, fmt.Errorf("%w: %w", ErrAnalysisFailed, errors.Join(preambleConstructionErrors...))
		}
		return cachedPreamble, nil
	}
	opLogger.Debug("Preamble not found in bbolt cache or cache invalid, generating...")

	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)
	preambleConstructionErrors = append(preambleConstructionErrors, loadErrors...)

	if targetPkg == nil {
		return "", fmt.Errorf("critical package load error for preamble: %w", errors.Join(preambleConstructionErrors...))
	}

	var cursorPos token.Pos = token.NoPos
	if targetFile != nil {
		var posErr error
		cursorPos, posErr = calculateCursorPos(targetFile, line, col, opLogger)
		if posErr != nil {
			preambleConstructionErrors = append(preambleConstructionErrors, fmt.Errorf("cannot calculate valid cursor position for preamble: %w", posErr))
		}
	} else {
		preambleConstructionErrors = append(preambleConstructionErrors, errors.New("target token.File is nil, cursor-specific context will be limited for preamble"))
	}

	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		CursorPos:      cursorPos,
		TargetPackage:  targetPkg,
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST,
		AnalysisErrors: make([]error, 0), // Collect errors specifically from preamble construction steps
		Diagnostics:    loadDiagnostics[:],
		PackageName:    targetPkg.Name,
	}
	// Add initial load errors to tempInfo's errors as well, so they are part of the context for preamble formatters
	tempInfo.AnalysisErrors = append(tempInfo.AnalysisErrors, loadErrors...)

	enclosingCtx, encErr := a.GetEnclosingContext(ctx, absFilename, version, line, col)
	if encErr != nil {
		addAnalysisError(tempInfo, fmt.Errorf("preamble: enclosing context error: %w", encErr), opLogger)
	}
	scopeInfo, scopeErr := a.GetScopeInfo(ctx, absFilename, version, line, col)
	if scopeErr != nil {
		addAnalysisError(tempInfo, fmt.Errorf("preamble: scope info error: %w", scopeErr), opLogger)
	}
	comments, commentErr := a.GetRelevantComments(ctx, absFilename, version, line, col)
	if commentErr != nil {
		addAnalysisError(tempInfo, fmt.Errorf("preamble: comments error: %w", commentErr), opLogger)
	}

	if tempInfo.TargetAstFile != nil && tempInfo.CursorPos.IsValid() {
		path, pathErr := findEnclosingPath(ctx, tempInfo.TargetAstFile, tempInfo.CursorPos, tempInfo, opLogger)
		if pathErr != nil {
			addAnalysisError(tempInfo, fmt.Errorf("preamble: finding path for cursor context failed: %w", pathErr), opLogger)
		}
		findContextNodes(ctx, path, tempInfo.CursorPos, tempInfo.TargetPackage, tempInfo.TargetFileSet, a, tempInfo, opLogger)
	} else if tempInfo.TargetAstFile == nil {
		addAnalysisError(tempInfo, errors.New("preamble: target AST is nil, cannot determine specific cursor context nodes"), opLogger)
	} else if !tempInfo.CursorPos.IsValid() {
		addAnalysisError(tempInfo, errors.New("preamble: cursor position invalid, cannot determine specific cursor context nodes"), opLogger)
	}
	// Errors from findContextNodes are added to tempInfo.AnalysisErrors directly.

	var qualifier types.Qualifier
	if targetPkg != nil && targetPkg.Types != nil {
		qualifier = types.RelativeTo(targetPkg.Types)
	} else {
		qualifier = func(other *types.Package) string {
			if other != nil {
				return other.Path()
			}
			return ""
		}
	}

	var preambleBuilder strings.Builder
	const internalPreambleLimit = 8192
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
		}
		msg := fmt.Sprintf("// ... (%s truncated)\n", section)
		if currentLen+len(msg) < internalPreambleLimit {
			preambleBuilder.WriteString(msg)
			currentLen += len(msg)
		}
		limitReached = true
	}

	pkgName := tempInfo.PackageName
	if pkgName == "" {
		pkgName = "[unknown]"
	}
	addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(absFilename), pkgName))

	if tempInfo.TargetAstFile != nil && !limitReached {
		tempInfo.Imports = tempInfo.TargetAstFile.Imports
		formatImportsSection(&preambleBuilder, tempInfo, addToPreamble, addTruncMarker, opLogger)
	}

	if !limitReached {
		formatEnclosingFuncSection(&preambleBuilder, enclosingCtx, qualifier, addToPreamble, opLogger)
	}
	if !limitReached {
		formatCommentsSection(&preambleBuilder, comments, addToPreamble, addTruncMarker, opLogger)
	}
	if !limitReached {
		formatCursorContextSection(&preambleBuilder, tempInfo, qualifier, addToPreamble, opLogger)
	}
	if !limitReached {
		formatScopeSection(&preambleBuilder, scopeInfo, qualifier, addToPreamble, addTruncMarker, opLogger)
	}

	generatedPreamble := preambleBuilder.String()
	if generatedPreamble == "" {
		opLogger.Warn("Preamble generation resulted in empty string.")
		if len(tempInfo.AnalysisErrors) > 0 || len(preambleConstructionErrors) > 0 {
			generatedPreamble = "// Warning: Preamble generation encountered errors and resulted in empty output.\n"
		}
	}

	// Add all errors collected in tempInfo during preamble specific steps to the main list
	preambleConstructionErrors = append(preambleConstructionErrors, tempInfo.AnalysisErrors...)

	if generatedPreamble != "" && a.db != nil && len(loadErrors) == 0 {
		saveStart := time.Now()
		inputHashes, hashErr := calculateInputHashes(dir, targetPkg, opLogger)
		if hashErr == nil {
			analysisDataToCache := CachedAnalysisData{PackageName: pkgName, PromptPreamble: generatedPreamble}
			var gobBuf bytes.Buffer
			if encodeErr := gob.NewEncoder(&gobBuf).Encode(&analysisDataToCache); encodeErr == nil {
				entryToSave := CachedAnalysisEntry{
					SchemaVersion: cacheSchemaVersion, GoModHash: goModHash, InputFileHashes: inputHashes, AnalysisGob: gobBuf.Bytes(),
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
		a.logger.Warn("SetMemoryCache failed (item might be too large or other constraint).", "key", key, "cost", cost, "ttl", ttl)
	}
	return set
}

// InvalidateCache removes the bbolt cached entry for a given directory.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	logger := a.logger.With("dir", dir, "op", "InvalidateCache")
	a.mu.RLock()
	db := a.db
	a.mu.RUnlock()

	if db == nil {
		logger.Debug("Bbolt cache invalidation skipped: DB is nil.")
		return nil
	}
	goModHash := calculateGoModHash(dir, logger)
	bboltCacheKey := []byte(dir + "::" + goModHash)

	logger.Info("Invalidating bbolt cache entry", "key", string(bboltCacheKey))
	return deleteCacheEntryByKey(db, bboltCacheKey, logger)
}

// InvalidateMemoryCacheForURI clears relevant entries from the ristretto memory cache.
func (a *GoPackagesAnalyzer) InvalidateMemoryCacheForURI(uri string, version int) error {
	logger := a.logger.With("uri", uri, "version", version, "op", "InvalidateMemoryCacheForURI")
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
	return nil
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
	maxFIMPartLen := maxSnippetLen / 2

	if snippetCtx.FallbackContext != "" {
		truncatedFallback := snippetCtx.FallbackContext
		if len(truncatedFallback) > maxPreambleLen {
			logger.Warn("Truncating fallback context", "original_length", len(truncatedFallback), "max_length", maxPreambleLen)
			marker := "// ... (Fallback context truncated)\n"
			startByte := len(truncatedFallback) - maxPreambleLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			if len(marker)+len(truncatedFallback[startByte:]) > maxPreambleLen {
				truncatedFallback = truncatedFallback[len(truncatedFallback)-maxPreambleLen:]
			} else {
				truncatedFallback = marker + truncatedFallback[startByte:]
			}
		}
		finalPreamble.WriteString(truncatedFallback)
		finalPreamble.WriteString("\n---\n")
	}

	truncatedPreamble := contextPreamble
	if len(truncatedPreamble) > maxPreambleLen {
		logger.Warn("Truncating context preamble", "original_length", len(truncatedPreamble), "max_length", maxPreambleLen)
		marker := "... (context truncated)\n"
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

	prefix := snippetCtx.Prefix
	suffix := snippetCtx.Suffix

	if config.UseFim {
		template := config.FimTemplate
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
	} else {
		template := config.PromptTemplate
		snippet := prefix
		if len(snippet) > maxSnippetLen {
			logger.Warn("Truncating code snippet (prefix)", "original_length", len(snippet), "max_length", maxSnippetLen)
			marker := "...(code truncated)\n"
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
	client    LLMClient
	analyzer  Analyzer
	formatter PromptFormatter
	config    Config
	configMu  sync.RWMutex
	logger    *stdslog.Logger
}

// NewDeepCompleter creates a new DeepCompleter service instance.
func NewDeepCompleter(logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
	serviceLogger := logger.With("service", "DeepCompleter")

	cfg, configErr := LoadConfig(serviceLogger)
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		serviceLogger.Error("Fatal error during initial config load", "error", configErr)
		return nil, configErr
	}
	if err := cfg.Validate(serviceLogger); err != nil {
		serviceLogger.Error("Initial configuration is invalid after loading/defaults", "error", err)
		if errors.Is(err, ErrInvalidConfig) {
			return nil, fmt.Errorf("initial config validation failed: %w", err)
		}
		serviceLogger.Warn("Initial config validation reported issues", "error", err)
		if configErr != nil {
			configErr = errors.Join(configErr, err)
		} else {
			configErr = err
		}
	}

	analyzer := NewGoPackagesAnalyzer(serviceLogger, cfg)
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    cfg,
		logger:    serviceLogger,
	}

	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with a specific config.
func NewDeepCompleterWithConfig(config Config, logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
	serviceLogger := logger.With("service", "DeepCompleter")

	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	if err := config.Validate(serviceLogger); err != nil {
		return nil, fmt.Errorf("provided config validation failed: %w", err)
	}

	analyzer := NewGoPackagesAnalyzer(serviceLogger, config)
	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    config,
		logger:    serviceLogger,
	}, nil
}

// Close cleans up resources used by the DeepCompleter.
func (dc *DeepCompleter) Close() error {
	dc.logger.Info("Closing DeepCompleter service")
	if dc.analyzer != nil {
		return dc.analyzer.Close()
	}
	return nil
}

// UpdateConfig atomically updates the completer's configuration.
func (dc *DeepCompleter) UpdateConfig(newConfig Config) error {
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate
	}
	if err := newConfig.Validate(dc.logger); err != nil {
		dc.logger.Error("Invalid configuration provided for update", "error", err)
		return fmt.Errorf("invalid configuration update: %w", err)
	}

	dc.configMu.Lock()
	dc.config = newConfig
	dc.configMu.Unlock()

	if dc.analyzer != nil {
		dc.analyzer.UpdateConfig(newConfig)
	}

	dc.logger.Info("DeepCompleter configuration updated",
		stdslog.Group("new_config",
			stdslog.String("ollama_url", newConfig.OllamaURL),
			stdslog.String("model", newConfig.Model),
			stdslog.Int("max_tokens", newConfig.MaxTokens),
			stdslog.Any("stop", newConfig.Stop),
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

// Client returns the LLMClient instance.
func (dc *DeepCompleter) Client() LLMClient {
	return dc.client
}

// InvalidateAnalyzerCache provides external access to invalidate the analyzer's disk cache.
func (dc *DeepCompleter) InvalidateAnalyzerCache(dir string) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	dc.logger.Info("Request to invalidate analyzer disk cache", "dir", dir)
	return dc.analyzer.InvalidateCache(dir)
}

// InvalidateMemoryCacheForURI provides external access to invalidate the analyzer's memory cache.
func (dc *DeepCompleter) InvalidateMemoryCacheForURI(uri string, version int) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	dc.logger.Info("Request to invalidate analyzer memory cache", "uri", uri, "version", version)
	return dc.analyzer.InvalidateMemoryCacheForURI(uri, version)
}

// GetAnalyzer returns the analyzer instance.
func (dc *DeepCompleter) GetAnalyzer() Analyzer {
	return dc.analyzer
}

// GetCompletion provides basic code completion for a given snippet without file context analysis.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	opLogger := dc.logger.With("operation", "GetCompletion")
	opLogger.Info("Handling basic completion request")
	currentConfig := dc.GetCurrentConfig()

	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet, Suffix: "", FullLine: ""}

	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig, opLogger)
	opLogger.Debug("Generated basic prompt", "length", len(prompt))

	var buffer bytes.Buffer
	apiCallFunc := func() error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()

		opLogger.Debug("Calling Ollama GenerateStream for basic completion")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig, opLogger)
		if apiErr != nil {
			return apiErr
		}
		defer reader.Close()

		streamCtx, cancelStream := context.WithTimeout(apiCtx, 50*time.Second)
		defer cancelStream()

		buffer.Reset()
		streamErr := streamCompletion(streamCtx, reader, &buffer, opLogger)
		if streamErr != nil {
			return fmt.Errorf("streaming completion failed: %w", streamErr)
		}
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, opLogger)
	if err != nil {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
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
	var contextPreamble string = "// Basic file context only."
	var preambleErr error

	if currentConfig.UseAst {
		opLogger.Info("Analyzing context for preamble (or checking cache)")
		preambleCtx, cancelPreamble := context.WithTimeout(ctx, 30*time.Second)
		contextPreamble, preambleErr = dc.analyzer.GetPromptPreamble(preambleCtx, absFilename, version, line, col)
		cancelPreamble()

		if preambleErr != nil && !errors.Is(preambleErr, ErrAnalysisFailed) {
			opLogger.Error("Fatal error getting prompt preamble", "error", preambleErr)
			return fmt.Errorf("failed to get prompt preamble: %w", preambleErr)
		}
		if preambleErr != nil {
			opLogger.Warn("Non-fatal error getting prompt preamble", "error", preambleErr)
			contextPreamble += fmt.Sprintf("\n// Warning: Preamble generation completed with errors: %v\n", preambleErr)
		}
		if contextPreamble == "" {
			contextPreamble = "// Warning: Context analysis returned empty preamble.\n"
			opLogger.Warn("Context analysis resulted in an empty preamble string.")
		}
	} else {
		opLogger.Info("AST analysis disabled by config. Using basic preamble.")
	}

	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col, currentConfig.UseAst, opLogger)
	if snippetErr != nil {
		opLogger.Error("Failed to extract code snippet context", "error", snippetErr)
		return fmt.Errorf("failed to extract code snippet context: %w", snippetErr)
	}

	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig, opLogger)
	opLogger.Debug("Generated prompt", "length", len(prompt))

	apiCallFunc := func() error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()
		opLogger.Debug("Calling Ollama GenerateStream")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig, opLogger)
		if apiErr != nil {
			return apiErr
		}
		defer reader.Close()
		streamErr := streamCompletion(apiCtx, reader, w, opLogger)
		if streamErr != nil {
			return fmt.Errorf("streaming completion failed: %w", streamErr)
		}
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, opLogger)
	if err != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
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

	if preambleErr != nil {
		opLogger.Warn("Completion stream successful, but context analysis for preamble encountered non-fatal errors", "analysis_error", preambleErr)
	} else {
		opLogger.Info("Completion stream successful")
	}
	return nil
}
