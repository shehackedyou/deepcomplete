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
	"golang.org/x/tools/go/packages" // Added for packages.Package type hint
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
	// TODO: Consider deprecating in favor of more specific methods below.
	Analyze(ctx context.Context, filename string, version int, line, col int) (*AstContextInfo, error)

	// GetIdentifierInfo attempts to find information about the identifier at the given position.
	GetIdentifierInfo(ctx context.Context, filename string, version int, line, col int) (*IdentifierInfo, error)

	// GetMemoryCache retrieves an item from the memory cache.
	GetMemoryCache(key string) (any, bool)
	// SetMemoryCache adds an item to the memory cache.
	SetMemoryCache(key string, value any, cost int64, ttl time.Duration) bool

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
		cfg = getDefaultConfig()
		logger.Info("Using default configuration values.")
	}

	finalCfg := cfg
	if err := finalCfg.Validate(logger); err != nil {
		logger.Error("Final configuration is invalid, falling back to pure defaults.", "error", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		pureDefault := getDefaultConfig()
		if valErr := pureDefault.Validate(logger); valErr != nil {
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
			Timeout: 90 * time.Second,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: 10 * time.Second,
				}).DialContext,
				TLSHandshakeTimeout:   10 * time.Second,
				MaxIdleConns:          10,
				IdleConnTimeout:       30 * time.Second,
				ResponseHeaderTimeout: 20 * time.Second,
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

	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second) // Short timeout for check
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

	// Any response (even error status codes like 404) means the server is reachable.
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

	opLogger.Debug("Sending generate request to Ollama", "url", endpointURL)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			opLogger.Warn("Ollama generate request context cancelled", "url", endpointURL)
			return nil, context.Canceled
		}
		if errors.Is(err, context.DeadlineExceeded) {
			opLogger.Error("Ollama generate request context deadline exceeded", "url", endpointURL, "timeout", c.httpClient.Timeout)
			return nil, fmt.Errorf("%w: context deadline exceeded: %w", ErrOllamaUnavailable, context.DeadlineExceeded)
		}

		var netErr net.Error
		if errors.As(err, &netErr) {
			if netErr.Timeout() {
				opLogger.Error("Network timeout during Ollama generate request", "host", u.Host, "error", netErr)
				return nil, fmt.Errorf("%w: network timeout: %w", ErrOllamaUnavailable, netErr)
			}
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
	mu          sync.Mutex       // Protects access to db/memoryCache handles during Close/Invalidate
	logger      *stdslog.Logger  // Stored logger instance
	config      Config           // Store config to access TTL easily
}

// NewGoPackagesAnalyzer initializes the analyzer, including setting up bbolt and ristretto caches.
func NewGoPackagesAnalyzer(logger *stdslog.Logger) *GoPackagesAnalyzer {
	// Note: We don't have the config here. The config should ideally be passed in,
	// or the TTL needs to be managed differently. For now, cache uses a constant TTL.
	// A better approach might be to have the DeepCompleter pass the TTL to SetMemoryCache.
	if logger == nil {
		logger = stdslog.Default()
	}
	analyzerLogger := logger.With("component", "GoPackagesAnalyzer")

	dbPath := ""
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil {
			dbPath = filepath.Join(dbDir, "analysis_cache.db")
		} else {
			analyzerLogger.Warn("Could not create bbolt cache directory, disk caching disabled.", "path", dbDir, "error", err)
		}
	} else {
		analyzerLogger.Warn("Could not determine user cache directory, disk caching disabled.", "error", err)
	}

	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second}
		db, err = bbolt.Open(dbPath, 0600, opts)
		if err != nil {
			analyzerLogger.Warn("Failed to open bbolt cache file, disk caching disabled.", "path", dbPath, "error", err)
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
				analyzerLogger.Warn("Failed to ensure bbolt bucket exists, disk caching disabled.", "error", err)
				db.Close()
				db = nil
			} else {
				analyzerLogger.Info("Using bbolt disk cache", "path", dbPath, "schema_version", cacheSchemaVersion)
			}
		}
	}

	memCache, cacheErr := ristretto.NewCache(&ristretto.Config{
		NumCounters: 1e7,
		MaxCost:     1 << 30, // 1GB
		BufferItems: 64,
		Metrics:     true,
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
		// config: config, // TODO: Pass config if needed for TTL or other settings
	}
}

// Close cleans up resources used by the analyzer.
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	var closeErrors []error
	logger := a.logger

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

// loadPackageForAnalysis handles the core logic of loading package data.
// This is extracted from the Analyze method for better modularity.
func (a *GoPackagesAnalyzer) loadPackageForAnalysis(ctx context.Context, absFilename string, fset *token.FileSet, logger *slog.Logger) (*packages.Package, *ast.File, *token.File, []Diagnostic, []error) {
	// This function reuses the logic previously in loadPackageAndFile
	// It's kept internal to GoPackagesAnalyzer for now.
	return loadPackageAndFile(ctx, absFilename, fset, logger)
}

// Analyze performs code analysis for a given file and position, utilizing caching.
// Returns the populated AstContextInfo and any fatal error encountered.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, absFilename string, version int, line, col int) (info *AstContextInfo, analysisErr error) {
	analysisLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "Analyze")

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
			analysisLogger.Error("Panic recovered during Analyze", "error", r, "stack", string(debug.Stack()))
			addAnalysisError(info, panicErr, analysisLogger)
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = fmt.Errorf("panic (%v) occurred after error: %w", r, analysisErr)
			}
		}
		// Consolidate non-fatal errors into ErrAnalysisFailed
		if len(info.AnalysisErrors) > 0 && analysisErr == nil {
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr)
		} else if len(info.AnalysisErrors) > 0 && analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			// If a fatal error occurred AND non-fatal errors, wrap them
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w (additional analysis issues: %w)", analysisErr, finalErr)
		}
	}()

	analysisLogger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)
	goModHash := calculateGoModHash(dir, analysisLogger)
	cacheKey := []byte(dir + "::" + goModHash)
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry
	var loadDuration, stepsDuration, preambleDuration time.Duration

	// --- Bbolt Cache Check (for preamble) ---
	if a.db != nil {
		// ... (bbolt cache check logic remains the same) ...
		readStart := time.Now()
		dbViewErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				return nil
			} // Bucket not found is not an error
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				return nil
			} // Cache miss

			var decoded CachedAnalysisEntry
			if err := gob.NewDecoder(bytes.NewReader(valBytes)).Decode(&decoded); err != nil {
				return fmt.Errorf("%w: failed to decode cache entry: %w", ErrCacheDecode, err)
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				analysisLogger.Warn("Bbolt cache data has old schema version. Ignoring.", "key", string(cacheKey), "cached_version", decoded.SchemaVersion, "expected_version", cacheSchemaVersion)
				return nil
			}
			cachedEntry = &decoded
			return nil
		})
		if dbViewErr != nil {
			analysisLogger.Warn("Error reading or decoding from bbolt cache.", "error", dbViewErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbViewErr), analysisLogger)
			if errors.Is(dbViewErr, ErrCacheDecode) {
				go deleteCacheEntryByKey(a.db, cacheKey, analysisLogger.With("reason", "decode_failure"))
			}
			cachedEntry = nil
		}
		analysisLogger.Debug("Bbolt cache read attempt finished", "duration", time.Since(readStart))

		if cachedEntry != nil {
			// ... (bbolt cache validation logic remains the same) ...
			validationStart := time.Now()
			currentHashes, hashErr := calculateInputHashes(dir, nil, analysisLogger)
			if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes, analysisLogger) {
				decodeStart := time.Now()
				var analysisData CachedAnalysisData
				if decodeErr := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob)).Decode(&analysisData); decodeErr == nil {
					info.PackageName = analysisData.PackageName
					info.PromptPreamble = analysisData.PromptPreamble
					cacheHit = true // Preamble cache hit
					loadDuration = time.Since(decodeStart)
					analysisLogger.Debug("Preamble successfully decoded from bbolt cache.", "duration", loadDuration)
					analysisLogger.Debug("Re-running load/analysis steps for diagnostics/hover despite cache hit for preamble.")
					cacheHit = false // Force re-analysis for other parts by resetting cacheHit flag for the *full* analysis
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
	info.TargetFileSet = fset
	// Use the extracted helper method
	targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, analysisLogger)
	info.Diagnostics = append(info.Diagnostics, loadDiagnostics...)
	loadDuration = time.Since(loadStart)
	analysisLogger.Debug("Package loading completed", "duration", loadDuration)
	for _, loadErr := range loadErrors {
		addAnalysisError(info, loadErr, analysisLogger) // Add non-fatal load errors
	}
	// Check for critical load failure
	if len(loadErrors) > 0 && targetPkg == nil {
		// analysisErr will be set by the defer func based on info.AnalysisErrors
		analysisLogger.Error("Critical package load error occurred.")
		return info, analysisErr // Return early if package load failed critically
	}
	info.TargetPackage = targetPkg
	info.TargetAstFile = targetFileAST
	// --- End Load Package Info ---

	// --- Perform Analysis Steps (using loaded info) ---
	stepsStart := time.Now()
	if targetFile != nil {
		// Pass the potentially loaded AST and package info to analysis steps
		analyzeStepErr := performAnalysisSteps(ctx, targetFile, targetFileAST, targetPkg, fset, line, col, a, info, analysisLogger)
		if analyzeStepErr != nil {
			// This error is likely a fatal cursor error from performAnalysisSteps
			analysisLogger.Error("Fatal error during performAnalysisSteps", "error", analyzeStepErr)
			// analysisErr will be set by the defer func
			return info, analyzeStepErr // Return the fatal error
		}
	} else {
		// Handle case where targetFile is nil (should be less likely now with earlier check)
		if len(loadErrors) == 0 { // Only log if no previous load error explains it
			addAnalysisError(info, errors.New("cannot perform analysis steps: target token.File is nil"), analysisLogger)
		}
		// Attempt to gather package scope even if file-specific analysis fails
		gatherScopeContext(ctx, nil, targetPkg, fset, info, analysisLogger)
	}
	stepsDuration = time.Since(stepsStart)
	analysisLogger.Debug("Analysis steps completed", "duration", stepsDuration)
	// --- End Analysis Steps ---

	// --- Construct Preamble (if not loaded from cache) ---
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
			analysisLogger.Debug("Building preamble with limited or no type info from targetPkg.")
		}
		info.PromptPreamble = constructPromptPreamble(a, info, qualifier, analysisLogger)
		preambleDuration = time.Since(preambleStart)
		analysisLogger.Debug("Preamble construction completed", "duration", preambleDuration)
	} else {
		analysisLogger.Debug("Skipping preamble construction as it was populated (e.g., from cache).")
	}
	// --- End Construct Preamble ---

	// --- Bbolt Cache Write (if preamble was generated) ---
	// Save preamble to cache only if it was *just* generated (not from cache hit)
	// and no critical load errors occurred.
	shouldSave := a.db != nil && info.PromptPreamble != "" && !cacheHit && len(loadErrors) == 0
	if shouldSave {
		// ... (bbolt cache write logic remains the same) ...
		analysisLogger.Debug("Attempting to save analysis results to bbolt cache.", "key", string(cacheKey))
		saveStart := time.Now()
		inputHashes, hashErr := calculateInputHashes(dir, targetPkg, analysisLogger)
		if hashErr == nil {
			analysisDataToCache := CachedAnalysisData{
				PackageName:    info.PackageName,
				PromptPreamble: info.PromptPreamble,
			}
			var gobBuf bytes.Buffer
			if encodeErr := gob.NewEncoder(&gobBuf).Encode(&analysisDataToCache); encodeErr == nil {
				analysisGob := gobBuf.Bytes()
				entryToSave := CachedAnalysisEntry{
					SchemaVersion: cacheSchemaVersion, GoModHash: goModHash, InputFileHashes: inputHashes, AnalysisGob: analysisGob,
				}
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
					analysisLogger.Warn("Failed to gob-encode cache entry", "error", entryEncodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr), analysisLogger)
				}
			} else {
				analysisLogger.Warn("Failed to gob-encode analysis data", "error", encodeErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr), analysisLogger)
			}
		} else {
			analysisLogger.Warn("Failed to calculate input hashes for cache save", "error", hashErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), analysisLogger)
		}
	} else if a.db != nil {
		analysisLogger.Debug("Skipping bbolt cache save", "key", string(cacheKey), "cache_hit", cacheHit, "load_errors", len(loadErrors), "preamble_empty", info.PromptPreamble == "")
	}
	// --- End Bbolt Cache Write ---

	analysisLogger.Info("Context analysis finished", "load_duration", loadDuration, "steps_duration", stepsDuration, "preamble_duration", preambleDuration)
	analysisLogger.Debug("Final Context Preamble generated", "length", len(info.PromptPreamble))
	analysisLogger.Debug("Final Diagnostics collected", "count", len(info.Diagnostics))

	// analysisErr is set by the defer func based on info.AnalysisErrors
	return info, analysisErr
}

// GetIdentifierInfo implements the Analyzer interface method.
// It focuses on finding the identifier at the cursor and its related type/definition info.
func (a *GoPackagesAnalyzer) GetIdentifierInfo(ctx context.Context, absFilename string, version int, line, col int) (*IdentifierInfo, error) {
	opLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "GetIdentifierInfo")
	opLogger.Info("Starting identifier analysis")

	// 1. Load Package and File AST (uses bbolt cache internally if configured)
	fset := token.NewFileSet()
	// Use internal helper
	targetPkg, targetFileAST, targetFile, _, loadErrors := a.loadPackageForAnalysis(ctx, absFilename, fset, opLogger)

	// Check for critical load errors
	if targetPkg == nil { // Critical failure if package couldn't be determined
		combinedErr := errors.Join(loadErrors...) // Combine any errors reported
		opLogger.Error("Critical error loading package, cannot get identifier info", "error", combinedErr)
		// Wrap the error for clarity
		return nil, fmt.Errorf("critical package load error: %w", combinedErr)
	}
	// Check if the specific file AST or token file is missing, even if package loaded
	if targetFile == nil || targetFileAST == nil {
		opLogger.Error("Target file AST or token.File is nil after loading, cannot get identifier info", "targetFile_nil", targetFile == nil, "targetAST_nil", targetFileAST == nil)
		// Combine load errors with this specific issue
		return nil, errors.Join(append(loadErrors, errors.New("target file AST or token.File is nil"))...)
	}
	// Check for missing type info, which is non-critical but limits results
	if targetPkg.TypesInfo == nil {
		opLogger.Warn("Type info is nil, identifier resolution might fail")
		// Proceed, but hover/definition might be incomplete
	}

	// 2. Calculate Cursor Position
	cursorPos, posErr := calculateCursorPos(targetFile, line, col, opLogger)
	if posErr != nil {
		// Wrap position calculation error
		return nil, fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		// Should be caught by calculateCursorPos, but double-check
		return nil, fmt.Errorf("%w: invalid cursor position calculated (Pos: %d)", ErrPositionConversion, cursorPos)
	}
	opLogger = opLogger.With("cursorPos", cursorPos, "cursorPosStr", fset.PositionFor(cursorPos, true).String())

	// 3. Find Path and Context Nodes (using helpers, potentially cached)
	tempInfo := &AstContextInfo{
		FilePath:       absFilename,
		Version:        version,
		CursorPos:      cursorPos,
		TargetPackage:  targetPkg,
		TargetFileSet:  fset,
		TargetAstFile:  targetFileAST,
		AnalysisErrors: make([]error, 0), // Initialize to capture non-fatal errors
	}

	path, pathErr := findEnclosingPath(ctx, targetFileAST, cursorPos, tempInfo, opLogger)
	if pathErr != nil {
		opLogger.Warn("Error finding enclosing path", "error", pathErr)
		// Add non-fatal error and continue if possible
		addAnalysisError(tempInfo, fmt.Errorf("finding enclosing path failed: %w", pathErr), opLogger)
	}
	// findContextNodes populates tempInfo.Identifier*, DefNode etc.
	findContextNodes(ctx, path, cursorPos, targetPkg, fset, a, tempInfo, opLogger)

	// Log any non-fatal errors accumulated during context node finding
	if len(tempInfo.AnalysisErrors) > 0 {
		opLogger.Warn("Non-fatal errors during context node finding", "errors", errors.Join(tempInfo.AnalysisErrors...))
	}

	// 4. Extract and Return Identifier Information
	if tempInfo.IdentifierAtCursor == nil || tempInfo.IdentifierObject == nil {
		opLogger.Debug("No identifier object found at cursor position")
		return nil, nil // Not finding an identifier isn't an error
	}

	// Read file content needed for range conversion later
	fileContent, readErr := os.ReadFile(absFilename)
	if readErr != nil {
		opLogger.Error("Failed to read file content for identifier info", "error", readErr)
		return nil, fmt.Errorf("failed to read file content %s: %w", absFilename, readErr)
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
	// Return found info. Non-fatal errors during analysis are logged but don't cause this specific function to fail.
	return identInfo, nil
}

// GetMemoryCache implements the Analyzer interface method.
func (a *GoPackagesAnalyzer) GetMemoryCache(key string) (any, bool) {
	a.mu.Lock()
	cache := a.memoryCache
	a.mu.Unlock()
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
	a.mu.Lock()
	cache := a.memoryCache
	a.mu.Unlock()
	if cache == nil {
		a.logger.Debug("SetMemoryCache skipped: Cache is nil.", "key", key)
		return false
	}
	set := cache.SetWithTTL(key, value, cost, ttl)
	if set {
		a.logger.Debug("SetMemoryCache success.", "key", key, "cost", cost, "ttl", ttl)
	} else {
		a.logger.Warn("SetMemoryCache failed.", "key", key, "cost", cost, "ttl", ttl)
	}
	return set
}

// InvalidateCache removes the bbolt cached entry for a given directory.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	logger := a.logger.With("dir", dir, "op", "InvalidateCache")
	a.mu.Lock()
	db := a.db
	a.mu.Unlock()

	if db == nil {
		logger.Debug("Bbolt cache invalidation skipped: DB is nil.")
		return nil
	}
	goModHash := calculateGoModHash(dir, logger)
	cacheKey := []byte(dir + "::" + goModHash)
	logger.Info("Invalidating bbolt cache entry", "key", string(cacheKey))
	return deleteCacheEntryByKey(db, cacheKey, logger)
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
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config, logger *stdslog.Logger) string {
	if logger == nil {
		logger = stdslog.Default()
	}
	var finalPreamble strings.Builder
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2

	// --- Add Fallback Context if present ---
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
		finalPreamble.WriteString("\n---\n") // Separator
	}

	// --- Add Main Preamble (AST-based or default) ---
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

	// --- Format Snippet based on FIM ---
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
	client    LLMClient       // Interface for interacting with the LLM backend.
	analyzer  Analyzer        // Interface for performing code analysis.
	formatter PromptFormatter // Interface for formatting the LLM prompt.
	config    Config          // Current active configuration
	configMu  sync.RWMutex    // Mutex to protect concurrent access to config.
	logger    *stdslog.Logger // Logger instance for the DeepCompleter service
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
	}

	analyzer := NewGoPackagesAnalyzer(serviceLogger)
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

	analyzer := NewGoPackagesAnalyzer(serviceLogger)
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
			stdslog.Int("memory_cache_ttl_seconds", newConfig.MemoryCacheTTLSeconds), // Log TTL
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
	// For basic completion, AST is not used, so pass false
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
	var contextPreamble string = "// Basic file context only." // Default preamble
	var analysisInfo *AstContextInfo
	var analysisErr error // Stores non-fatal analysis errors (ErrAnalysisFailed)

	if currentConfig.UseAst {
		opLogger.Info("Analyzing context (or checking cache)")
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second)
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, absFilename, version, line, col)
		cancelAnalysis()

		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			opLogger.Error("Fatal error during analysis/cache check", "error", analysisErr)
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		if analysisErr != nil {
			opLogger.Warn("Non-fatal error during analysis/cache check", "error", analysisErr)
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
		opLogger.Info("AST analysis disabled by config.")
	}

	// Pass currentConfig.UseAst to extractSnippetContext
	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col, currentConfig.UseAst, opLogger)
	if snippetErr != nil {
		opLogger.Error("Failed to extract code snippet context", "error", snippetErr)
		return fmt.Errorf("failed to extract code snippet context: %w", snippetErr)
	}

	// Note: Formatter now handles prepending FallbackContext if present.
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

	if analysisErr != nil {
		opLogger.Warn("Completion stream successful, but context analysis encountered non-fatal errors", "analysis_error", analysisErr)
	} else {
		opLogger.Info("Completion stream successful")
	}
	return nil
}
