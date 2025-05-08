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
// Global Variables (Package Level)
// =============================================================================

var (
	cacheBucketName = []byte("AnalysisCache") // Name of the bbolt bucket for caching.
)

// Core type definitions (Config, AstContextInfo, Diagnostic, etc.) are in deepcomplete_types.go.
// Exported error variables (Err*) are in deepcomplete_errors.go.

// =============================================================================
// Interfaces for Components
// =============================================================================

// LLMClient defines the interface for interacting with the language model backend.
type LLMClient interface {
	// GenerateStream sends a prompt to the LLM and returns a stream of generated text.
	GenerateStream(ctx context.Context, prompt string, config Config, logger *stdslog.Logger) (io.ReadCloser, error)
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
	FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config, logger *stdslog.Logger) string
}

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadConfig loads configuration from standard locations, merges with defaults,
// validates, and attempts to write a default config if needed.
// Returns the final Config and a non-fatal ErrConfig if warnings occurred.
func LoadConfig(logger *stdslog.Logger) (Config, error) {
	if logger == nil {
		logger = stdslog.Default() // Use default if nil
	}
	cfg := getDefaultConfig() // Start with defaults (from types.go)
	var loadedFromFile bool
	var loadErrors []error
	var configParseError error

	primaryPath, secondaryPath, pathErr := GetConfigPaths(logger)
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		logger.Warn("Could not determine config paths, using defaults", "error", pathErr)
	}

	// Try primary path
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

	// Try secondary path if primary failed or wasn't found
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
		} else if loaded && !loadedFromFile { // Only mark loadedFromFile if it wasn't already from primary
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
			if err := WriteDefaultConfig(writePath, getDefaultConfig(), logger); err != nil {
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
			Timeout: 90 * time.Second, // Overall request timeout
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: 10 * time.Second, // Connection timeout
				}).DialContext,
				TLSHandshakeTimeout:   10 * time.Second,
				MaxIdleConns:          10,
				IdleConnTimeout:       30 * time.Second,
				ResponseHeaderTimeout: 20 * time.Second, // Timeout for receiving headers
			},
		},
	}
}

// GenerateStream sends a request to Ollama's /api/generate endpoint and returns the streaming response body.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config, logger *stdslog.Logger) (io.ReadCloser, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
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
			"num_ctx":     4096, // Consider making this configurable or deriving it
			"top_p":       0.9,  // Common default, consider exposing
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
	req.Header.Set("Accept", "application/x-ndjson") // Ollama streams ndjson

	logger.Debug("Sending request to Ollama", "url", endpointURL, "model", config.Model)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Handle context cancellation or deadline explicitly
		if errors.Is(err, context.Canceled) {
			logger.Warn("Ollama request cancelled", "url", endpointURL)
			return nil, context.Canceled
		}
		if errors.Is(err, context.DeadlineExceeded) {
			logger.Error("Ollama request timed out", "url", endpointURL, "timeout", c.httpClient.Timeout)
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		// Check for network errors (e.g., connection refused)
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			logger.Error("Network timeout connecting to Ollama", "host", u.Host, "error", netErr)
			return nil, fmt.Errorf("%w: network timeout connecting to %s: %w", ErrOllamaUnavailable, u.Host, netErr)
		}
		if opErr, ok := err.(*net.OpError); ok && opErr.Op == "dial" {
			logger.Error("Connection refused or network error connecting to Ollama", "host", u.Host, "error", opErr)
			return nil, fmt.Errorf("%w: connection refused or network error connecting to %s: %w", ErrOllamaUnavailable, u.Host, opErr)
		}
		// Generic HTTP error
		logger.Error("HTTP request to Ollama failed", "url", endpointURL, "error", err)
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
		// Use the custom OllamaError type
		err = &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		logger.Error("Ollama API returned non-OK status", "status", resp.Status, "response_body", bodyString)
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
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
}

// NewGoPackagesAnalyzer initializes the analyzer, including setting up bbolt and ristretto caches.
func NewGoPackagesAnalyzer(logger *stdslog.Logger) *GoPackagesAnalyzer {
	if logger == nil {
		logger = stdslog.Default() // Fallback to default logger if nil
	}
	analyzerLogger := logger.With("component", "GoPackagesAnalyzer") // Add component context

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
		NumCounters: 1e7,     // 10M
		MaxCost:     1 << 30, // 1GB
		BufferItems: 64,      // Default
		Metrics:     true,    // Enable metrics
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
		logger:      analyzerLogger, // Store the logger
	}
}

// Close cleans up resources used by the analyzer.
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	var closeErrors []error
	// Use the stored logger instance
	logger := a.logger

	if a.db != nil {
		logger.Info("Closing bbolt cache database.")
		if err := a.db.Close(); err != nil {
			logger.Error("Error closing bbolt database", "error", err)
			closeErrors = append(closeErrors, fmt.Errorf("bbolt close failed: %w", err))
		}
		a.db = nil // Mark as closed
	}
	if a.memoryCache != nil {
		logger.Info("Closing ristretto memory cache.")
		a.memoryCache.Close()
		a.memoryCache = nil // Mark as closed
	}

	if len(closeErrors) > 0 {
		return errors.Join(closeErrors...)
	}
	return nil
}

// Analyze performs code analysis for a given file and position, utilizing caching.
// Returns the populated AstContextInfo and any fatal error encountered.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, absFilename string, version int, line, col int) (info *AstContextInfo, analysisErr error) {
	// Use the stored logger instance, add context for this specific analysis run
	analysisLogger := a.logger.With("absFile", absFilename, "version", version, "line", line, "col", col, "op", "Analyze")

	info = &AstContextInfo{
		FilePath:         absFilename,
		Version:          version,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		Diagnostics:      make([]Diagnostic, 0),
		CallArgIndex:     -1, // Initialize to -1 (no call context)
	}

	// Panic recovery using the stored logger
	defer func() {
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			analysisLogger.Error("Panic recovered during Analyze", "error", r, "stack", string(debug.Stack()))
			// Use helper from helpers_diagnostics.go, pass the specific logger
			addAnalysisError(info, panicErr, analysisLogger)
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				// Wrap existing error with panic info
				analysisErr = fmt.Errorf("panic (%v) occurred after error: %w", r, analysisErr)
			}
		}
		// Consolidate analysis errors if any occurred
		if len(info.AnalysisErrors) > 0 && analysisErr == nil {
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr)
		} else if len(info.AnalysisErrors) > 0 && analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			// If analysisErr is already set (and not just ErrAnalysisFailed), join with collected errors
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w (additional analysis issues: %w)", analysisErr, finalErr)
		}
	}()

	analysisLogger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)
	// Pass logger to utility functions
	goModHash := calculateGoModHash(dir, analysisLogger)
	cacheKey := []byte(dir + "::" + goModHash)
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry
	var loadDuration, stepsDuration, preambleDuration time.Duration

	// --- Bbolt Cache Check ---
	if a.db != nil {
		readStart := time.Now()
		dbViewErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				// Bucket not existing is not an error itself, just means no cache yet
				analysisLogger.Debug("Bbolt cache bucket does not exist.", "bucket_name", string(cacheBucketName))
				return nil
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				analysisLogger.Debug("Bbolt cache miss.", "key", string(cacheKey))
				return nil
			}

			analysisLogger.Debug("Bbolt cache hit (raw bytes found). Decoding entry...", "key", string(cacheKey))
			var decoded CachedAnalysisEntry
			if err := gob.NewDecoder(bytes.NewReader(valBytes)).Decode(&decoded); err != nil {
				// Return wrapped error for better context
				return fmt.Errorf("%w: failed to decode cache entry: %w", ErrCacheDecode, err)
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				analysisLogger.Warn("Bbolt cache data has old schema version. Ignoring.", "key", string(cacheKey), "cached_version", decoded.SchemaVersion, "expected_version", cacheSchemaVersion)
				return nil // Treat as miss
			}
			cachedEntry = &decoded
			return nil
		})
		if dbViewErr != nil {
			analysisLogger.Warn("Error reading or decoding from bbolt cache. Cache check failed.", "error", dbViewErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbViewErr), analysisLogger)
			if errors.Is(dbViewErr, ErrCacheDecode) {
				// Pass logger to utility function
				go deleteCacheEntryByKey(a.db, cacheKey, analysisLogger.With("reason", "decode_failure"))
			}
			cachedEntry = nil // Ensure it's nil if there was an error
		}
		analysisLogger.Debug("Bbolt cache read attempt finished", "duration", time.Since(readStart))

		// --- Bbolt Cache Validation ---
		if cachedEntry != nil {
			validationStart := time.Now()
			analysisLogger.Debug("Potential bbolt cache hit. Validating file hashes...", "key", string(cacheKey))
			// Pass logger to utility functions
			currentHashes, hashErr := calculateInputHashes(dir, nil, analysisLogger)
			if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes, analysisLogger) {
				analysisLogger.Debug("Bbolt cache VALID. Attempting to decode analysis data...", "key", string(cacheKey))
				decodeStart := time.Now()
				var analysisData CachedAnalysisData
				if decodeErr := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob)).Decode(&analysisData); decodeErr == nil {
					info.PackageName = analysisData.PackageName
					info.PromptPreamble = analysisData.PromptPreamble
					cacheHit = true // Mark as hit
					loadDuration = time.Since(decodeStart)
					analysisLogger.Debug("Analysis data successfully decoded from bbolt cache.", "duration", loadDuration)
					// Forcing re-analysis for diagnostics and hover even on cache hit for preamble
					analysisLogger.Debug("Re-running load/analysis steps for diagnostics/hover despite cache hit for preamble.")
					cacheHit = false // Force re-analysis for other parts
				} else {
					analysisLogger.Warn("Failed to gob-decode cached analysis data. Treating as miss.", "error", decodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheDecode, decodeErr), analysisLogger)
					// Pass logger to utility function
					go deleteCacheEntryByKey(a.db, cacheKey, analysisLogger.With("reason", "analysis_decode_failure"))
					cacheHit = false
				}
			} else {
				analysisLogger.Debug("Bbolt cache INVALID (hash mismatch or error). Treating as miss.", "key", string(cacheKey), "hash_error", hashErr)
				// Pass logger to utility function
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

	// --- Perform Full Analysis if Cache Miss or Invalid ---
	if !cacheHit {
		analysisLogger.Debug("Cache miss or invalid. Performing full analysis...", "key", string(cacheKey))
		loadStart := time.Now()
		fset := token.NewFileSet()
		info.TargetFileSet = fset // Store fileset in info

		var loadDiagnostics []Diagnostic
		var loadErrors []error
		// Pass context and the specific analysisLogger down
		targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := loadPackageAndFile(ctx, absFilename, fset, analysisLogger)
		info.Diagnostics = append(info.Diagnostics, loadDiagnostics...)

		loadDuration = time.Since(loadStart)
		analysisLogger.Debug("packages.Load completed", "duration", loadDuration)
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr, analysisLogger)
		}
		info.TargetPackage = targetPkg
		info.TargetAstFile = targetFileAST

		stepsStart := time.Now()
		if targetFile != nil { // Ensure targetFile (token.File) is not nil
			// Pass context and analysisLogger down
			analyzeStepErr := performAnalysisSteps(ctx, targetFile, targetFileAST, targetPkg, fset, line, col, a, info, analysisLogger)
			if analyzeStepErr != nil {
				// This error is already added to info.AnalysisErrors by performAnalysisSteps or its callees
				// if it's non-fatal, or returned if fatal (like invalid cursor).
				// If it's fatal, it will be caught by the defer func.
				analysisLogger.Error("Error during performAnalysisSteps", "error", analyzeStepErr)
			}
		} else {
			// If targetFile is nil (e.g., from loadPackageAndFile error), we can't do file-specific analysis
			if len(loadErrors) == 0 { // Only add this error if no prior load error explains it
				addAnalysisError(info, errors.New("cannot perform analysis steps: target token.File is nil (likely due to load error)"), analysisLogger)
			}
			// Still attempt to gather package-level scope context
			gatherScopeContext(ctx, nil, targetPkg, fset, info, analysisLogger)
		}
		stepsDuration = time.Since(stepsStart)
		analysisLogger.Debug("Analysis steps completed", "duration", stepsDuration)

		// Construct preamble only if it wasn't loaded from cache (or cache hit was overridden)
		if info.PromptPreamble == "" {
			preambleStart := time.Now()
			var qualifier types.Qualifier
			if targetPkg != nil && targetPkg.Types != nil {
				if info.PackageName == "" { // Set if not already set
					info.PackageName = targetPkg.Types.Name()
				}
				qualifier = types.RelativeTo(targetPkg.Types)
			} else {
				qualifier = func(other *types.Package) string { // Fallback qualifier
					if other != nil {
						return other.Path() // Use path for unknown packages
					}
					return ""
				}
				analysisLogger.Debug("Building preamble with limited or no type info from targetPkg.")
			}
			// Pass analysisLogger down
			info.PromptPreamble = constructPromptPreamble(a, info, qualifier, analysisLogger)
			preambleDuration = time.Since(preambleStart)
			analysisLogger.Debug("Preamble construction completed", "duration", preambleDuration)
		} else {
			analysisLogger.Debug("Skipping preamble construction as it was populated (e.g. from cache, though re-analysis was forced).")
		}

		// --- Bbolt Cache Write ---
		// Save to cache if DB is available, preamble was generated, and no critical load errors occurred
		shouldSave := a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0
		if shouldSave {
			analysisLogger.Debug("Attempting to save analysis results to bbolt cache.", "key", string(cacheKey))
			saveStart := time.Now()
			// Pass analysisLogger to utility function
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
						SchemaVersion:   cacheSchemaVersion,
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
								return fmt.Errorf("%w: cache bucket %s disappeared", ErrCacheWrite, string(cacheBucketName))
							}
							analysisLogger.Debug("Writing bytes to bbolt cache", "key", string(cacheKey), "bytes", len(encodedBytes))
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
			analysisLogger.Debug("Skipping bbolt cache save", "key", string(cacheKey), "load_errors", len(loadErrors), "preamble_empty", info.PromptPreamble == "")
		}
	} // End if !cacheHit

	if cacheHit { // This condition is now less likely to be true due to forced re-analysis
		analysisLogger.Info("Context analysis finished (cache hit for preamble, re-analyzed for diagnostics)", "decode_duration", loadDuration)
	} else {
		analysisLogger.Info("Context analysis finished (full analysis or re-analysis after cache)", "load_duration", loadDuration, "steps_duration", stepsDuration, "preamble_duration", preambleDuration)
	}
	analysisLogger.Debug("Final Context Preamble generated", "length", len(info.PromptPreamble))
	analysisLogger.Debug("Final Diagnostics collected", "count", len(info.Diagnostics))

	return info, analysisErr
}

// InvalidateCache removes the bbolt cached entry for a given directory.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	// Use the stored logger instance
	logger := a.logger.With("dir", dir, "op", "InvalidateCache")
	a.mu.Lock()
	db := a.db // Access under lock
	a.mu.Unlock()

	if db == nil {
		logger.Debug("Bbolt cache invalidation skipped: DB is nil.")
		return nil
	}
	// Pass logger to utility function
	goModHash := calculateGoModHash(dir, logger)
	cacheKey := []byte(dir + "::" + goModHash)
	logger.Info("Invalidating bbolt cache entry", "key", string(cacheKey))
	// Pass logger to utility function
	return deleteCacheEntryByKey(db, cacheKey, logger)
}

// InvalidateMemoryCacheForURI clears relevant entries from the ristretto memory cache.
func (a *GoPackagesAnalyzer) InvalidateMemoryCacheForURI(uri string, version int) error {
	// Use the stored logger instance
	logger := a.logger.With("uri", uri, "version", version, "op", "InvalidateMemoryCacheForURI")
	a.mu.Lock()
	memCache := a.memoryCache // Access under lock
	a.mu.Unlock()

	if memCache == nil {
		logger.Debug("Memory cache invalidation skipped: Cache is nil.")
		return nil
	}
	// Current strategy is to clear the whole memory cache on any document change.
	// This is simple but could be refined for more granular invalidation if performance dictates.
	logger.Warn("Clearing entire Ristretto memory cache due to document change.", "uri", uri)
	memCache.Clear()
	// Consider logging metrics after clear if useful: logger.Debug("Memory cache cleared", "metrics", memCache.Metrics)
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
	var finalPrompt strings.Builder // Use strings.Builder for efficiency
	template := config.PromptTemplate
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2 // Ensure this is reasonable

	// Truncate context preamble
	if len(contextPreamble) > maxPreambleLen {
		logger.Warn("Truncating context preamble", "original_length", len(contextPreamble), "max_length", maxPreambleLen)
		marker := "... (context truncated)\n"
		startByte := len(contextPreamble) - maxPreambleLen + len(marker)
		if startByte < 0 { // Ensure startByte is not negative
			startByte = 0
		}
		// Ensure marker + truncated part does not exceed maxPreambleLen
		if len(marker)+len(contextPreamble[startByte:]) > maxPreambleLen {
			// If it does, just take the end of the preamble
			contextPreamble = contextPreamble[len(contextPreamble)-maxPreambleLen:]
		} else {
			contextPreamble = marker + contextPreamble[startByte:]
		}
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
			// Ensure marker + truncated part fits
			if len(marker)+len(prefix[startByte:]) > maxFIMPartLen {
				prefix = prefix[len(prefix)-maxFIMPartLen:]
			} else {
				prefix = marker + prefix[startByte:]
			}
		}
		// Truncate FIM suffix
		if len(suffix) > maxFIMPartLen {
			logger.Warn("Truncating FIM suffix", "original_length", len(suffix), "max_length", maxFIMPartLen)
			marker := "(suffix truncated)..."
			endByte := maxFIMPartLen - len(marker)
			if endByte < 0 {
				endByte = 0
			}
			// Ensure truncated part + marker fits
			if len(suffix[:endByte])+len(marker) > maxFIMPartLen {
				suffix = suffix[:maxFIMPartLen]
			} else {
				suffix = suffix[:endByte] + marker
			}
		}
		// Use fmt.Fprintf for safe formatting into the builder
		fmt.Fprintf(&finalPrompt, template, contextPreamble, prefix, suffix)
	} else {
		snippet := snippetCtx.Prefix // For non-FIM, the snippet is just the prefix
		// Truncate snippet (prefix)
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
		fmt.Fprintf(&finalPrompt, template, contextPreamble, snippet)
	}
	return finalPrompt.String()
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
// It initializes components and loads the initial configuration.
func NewDeepCompleter(logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default() // Fallback if no logger provided
	}
	serviceLogger := logger.With("service", "DeepCompleter")

	cfg, configErr := LoadConfig(serviceLogger) // Pass serviceLogger for config loading
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		// This is a fatal error if not ErrConfig (which indicates non-fatal warnings)
		serviceLogger.Error("Fatal error during initial config load", "error", configErr)
		return nil, configErr
	}
	// Validate even if there were non-fatal ErrConfig issues
	if err := cfg.Validate(serviceLogger); err != nil {
		// If validation fails after loading (or with defaults), it's problematic.
		serviceLogger.Error("Initial configuration is invalid after loading/defaults", "error", err)
		// Depending on severity, might return error or proceed with a known safe default.
		// For now, if Validate returns ErrInvalidConfig, it's fatal.
		if errors.Is(err, ErrInvalidConfig) {
			return nil, fmt.Errorf("initial config validation failed: %w", err)
		}
		// Log other validation warnings
		serviceLogger.Warn("Initial config validation reported issues", "error", err)
	}

	// Initialize analyzer, passing the serviceLogger
	analyzer := NewGoPackagesAnalyzer(serviceLogger)
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    cfg,
		logger:    serviceLogger, // Store the service-level logger
	}

	// Return with ErrConfig if it occurred, allowing caller to know about warnings
	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with a specific config.
// This is useful for testing or specific programmatic setups.
func NewDeepCompleterWithConfig(config Config, logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default()
	}
	serviceLogger := logger.With("service", "DeepCompleter")

	// Ensure internal templates are set if not provided in config
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	if err := config.Validate(serviceLogger); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	analyzer := NewGoPackagesAnalyzer(serviceLogger) // Pass logger
	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    config,
		logger:    serviceLogger,
	}, nil
}

// Close cleans up resources used by the DeepCompleter, primarily the analyzer.
func (dc *DeepCompleter) Close() error {
	dc.logger.Info("Closing DeepCompleter service")
	if dc.analyzer != nil {
		return dc.analyzer.Close()
	}
	return nil
}

// UpdateConfig atomically updates the completer's configuration.
// It validates the new configuration before applying it.
func (dc *DeepCompleter) UpdateConfig(newConfig Config) error {
	// Ensure internal templates are always set
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate
	}
	// Validate the new configuration using the service's logger
	if err := newConfig.Validate(dc.logger); err != nil {
		dc.logger.Error("Invalid configuration provided for update", "error", err)
		return fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	dc.configMu.Lock()
	dc.config = newConfig
	dc.configMu.Unlock()

	// Log the updated configuration details
	dc.logger.Info("DeepCompleter configuration updated",
		stdslog.Group("new_config",
			stdslog.String("ollama_url", newConfig.OllamaURL),
			stdslog.String("model", newConfig.Model),
			stdslog.Int("max_tokens", newConfig.MaxTokens),
			stdslog.Any("stop", newConfig.Stop), // slog.Any handles slices
			stdslog.Float64("temperature", newConfig.Temperature),
			stdslog.String("log_level", newConfig.LogLevel),
			stdslog.Bool("use_ast", newConfig.UseAst),
			stdslog.Bool("use_fim", newConfig.UseFim),
			stdslog.Int("max_preamble_len", newConfig.MaxPreambleLen),
			stdslog.Int("max_snippet_len", newConfig.MaxSnippetLen),
		),
	)
	return nil
}

// GetCurrentConfig returns a thread-safe copy of the current configuration.
func (dc *DeepCompleter) GetCurrentConfig() Config {
	dc.configMu.RLock()
	defer dc.configMu.RUnlock()
	// Create a copy to avoid external modification of the internal config
	cfgCopy := dc.config
	// Ensure slices are also copied if they are part of the config
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

// GetAnalyzer returns the analyzer instance, useful for direct cache interaction or metrics.
func (dc *DeepCompleter) GetAnalyzer() Analyzer {
	return dc.analyzer
}

// GetCompletion provides basic code completion for a given snippet without file context analysis.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	// Use the service logger, add operation-specific context
	opLogger := dc.logger.With("operation", "GetCompletion")
	opLogger.Info("Handling basic completion request")
	currentConfig := dc.GetCurrentConfig()

	// Basic preamble for non-AST context
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet, Suffix: "", FullLine: ""} // Suffix/FullLine not used for basic

	// Pass opLogger to formatter
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig, opLogger)
	opLogger.Debug("Generated basic prompt", "length", len(prompt))

	var buffer bytes.Buffer
	apiCallFunc := func() error {
		select {
		case <-ctx.Done():
			return ctx.Err() // Respect outer context cancellation
		default:
		}
		// Create a new context for the API call with its own timeout
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()

		opLogger.Debug("Calling Ollama GenerateStream for basic completion")
		// Pass opLogger to client's GenerateStream
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig, opLogger)
		if apiErr != nil {
			return apiErr // Propagate error from LLM client
		}
		defer reader.Close() // Ensure stream body is closed

		// Context for stream processing, slightly shorter than API context
		streamCtx, cancelStream := context.WithTimeout(apiCtx, 50*time.Second)
		defer cancelStream()

		buffer.Reset() // Ensure buffer is clean for this attempt
		// Pass opLogger to streamCompletion utility
		streamErr := streamCompletion(streamCtx, reader, &buffer, opLogger)
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		return nil
	}

	// Pass opLogger to retry utility
	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, opLogger)
	if err != nil {
		select {
		case <-ctx.Done(): // Check if the original context was cancelled
			return "", ctx.Err()
		default:
		}
		// Handle specific error types after retries
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) {
			opLogger.Error("Ollama unavailable for basic completion after retries", "error", err)
			return "", fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			opLogger.Error("Stream processing error for basic completion after retries", "error", err)
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
	var analysisErr error // To store non-fatal analysis errors

	if currentConfig.UseAst {
		opLogger.Info("Analyzing context (or checking cache)")
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second) // Timeout for analysis
		// Analyze method uses its own configured logger
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, absFilename, version, line, col)
		cancelAnalysis() // Release resources if analysis finishes early

		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			// This is a fatal analysis error (not just ErrAnalysisFailed)
			opLogger.Error("Fatal error during analysis/cache check", "error", analysisErr)
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		// If ErrAnalysisFailed, it means there were non-fatal issues.
		// We can still proceed but should note the errors.
		if analysisErr != nil { // This implies errors.Is(analysisErr, ErrAnalysisFailed)
			opLogger.Warn("Non-fatal error during analysis/cache check", "error", analysisErr)
			if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
				// Append warning to the existing preamble
				contextPreamble = analysisInfo.PromptPreamble + fmt.Sprintf("\n// Warning: Context analysis completed with errors: %v\n", analysisErr)
			} else {
				// Use default preamble and append warning
				contextPreamble += fmt.Sprintf("\n// Warning: Context analysis completed with errors: %v\n", analysisErr)
			}
		} else if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else {
			// Analysis succeeded but no specific preamble was generated
			contextPreamble += "\n// Warning: Context analysis returned no specific context preamble.\n"
		}
	} else {
		opLogger.Info("AST analysis disabled by config.")
	}

	// Pass opLogger to utility function
	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col, opLogger)
	if snippetErr != nil {
		opLogger.Error("Failed to extract code snippet context", "error", snippetErr)
		return fmt.Errorf("failed to extract code snippet context: %w", snippetErr)
	}

	// Pass opLogger to formatter
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
		// Pass opLogger to client's GenerateStream
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig, opLogger)
		if apiErr != nil {
			return apiErr
		}
		defer reader.Close()
		// Pass opLogger to streamCompletion utility
		streamErr := streamCompletion(apiCtx, reader, w, opLogger)
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		return nil
	}

	// Pass opLogger to retry utility
	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, opLogger)
	if err != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) {
			opLogger.Error("Ollama unavailable for stream after retries", "error", err)
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			opLogger.Error("Stream processing error for stream after retries", "error", err)
			return err
		}
		opLogger.Error("Failed to get completion stream after retries", "error", err)
		return fmt.Errorf("failed to get completion stream after %d retries: %w", maxRetries, err)
	}

	if analysisErr != nil { // Log if analysis had non-fatal issues but completion succeeded
		opLogger.Warn("Completion stream successful, but context analysis encountered non-fatal errors", "analysis_error", analysisErr)
	} else {
		opLogger.Info("Completion stream successful")
	}
	return nil
}
