// deepcomplete.go
// Package deepcomplete provides core logic for local code completion using LLMs.
// Cycle 1: Moved core types and errors to separate files.
// Cycle 2: Moved utility functions to deepcomplete_utils.go. Component implementations remain here for now.
// Cycle 2 Fix: Moved MemoryCacheEnabled and GetMemoryCacheMetrics back here as methods. Updated calls to exported config utils.
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
	MemoryCacheEnabled() bool // Cycle 2 Fix: Method defined on GoPackagesAnalyzer
	// GetMemoryCacheMetrics returns performance metrics for the in-memory cache.
	GetMemoryCacheMetrics() *ristretto.Metrics // Cycle 2 Fix: Method defined on GoPackagesAnalyzer
}

// PromptFormatter defines the interface for constructing the final prompt sent to the LLM.
type PromptFormatter interface {
	// FormatPrompt combines the analysis preamble and code snippet context into the final prompt string.
	FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string
}

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadConfig loads configuration from standard locations, merges with defaults,
// validates, and attempts to write a default config if needed.
// Returns the final Config and a non-fatal ErrConfig if warnings occurred.
// Cycle 2 Fix: Calls exported utility functions from deepcomplete_utils.go.
func LoadConfig(logger *stdslog.Logger) (Config, error) {
	if logger == nil {
		logger = stdslog.Default() // Use default if nil
	}
	cfg := getDefaultConfig() // Start with defaults (from types.go)
	var loadedFromFile bool
	var loadErrors []error
	var configParseError error

	// Use exported function from utils
	primaryPath, secondaryPath, pathErr := GetConfigPaths(logger)
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		logger.Warn("Could not determine config paths, using defaults", "error", pathErr)
	}

	// Try primary path
	if primaryPath != "" {
		logger.Debug("Attempting to load config", "path", primaryPath)
		// Use exported function from utils
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
		// Use exported function from utils
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
			// Use exported function from utils
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

// =============================================================================
// Default Component Implementations
// =============================================================================
// Cycle 2: Kept implementations here for now, might refactor later.

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
				TLSHandshakeTimeout: 10 * time.Second,
				MaxIdleConns:        10,
				IdleConnTimeout:     30 * time.Second,
			},
		},
	}
}

// GenerateStream sends a request to Ollama's /api/generate endpoint and returns the streaming response body.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	logger := stdslog.Default()
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

	logger.Debug("Sending request to Ollama", "url", endpointURL, "model", config.Model)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			logger.Warn("Ollama request cancelled", "url", endpointURL)
			return nil, context.Canceled
		}
		if errors.Is(err, context.DeadlineExceeded) {
			logger.Error("Ollama request timed out", "url", endpointURL, "timeout", c.httpClient.Timeout)
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			logger.Error("Network timeout connecting to Ollama", "host", u.Host)
			return nil, fmt.Errorf("%w: network timeout connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		if opErr, ok := err.(*net.OpError); ok && opErr.Op == "dial" {
			logger.Error("Connection refused or network error connecting to Ollama", "host", u.Host)
			return nil, fmt.Errorf("%w: connection refused or network error connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		logger.Error("HTTP request to Ollama failed", "url", endpointURL, "error", err)
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
}

// NewGoPackagesAnalyzer initializes the analyzer, including setting up bbolt and ristretto caches.
func NewGoPackagesAnalyzer(logger *stdslog.Logger) *GoPackagesAnalyzer {
	if logger == nil {
		logger = stdslog.Default()
	}
	dbPath := ""

	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil {
			dbPath = filepath.Join(dbDir, "analysis_cache.db")
		} else {
			logger.Warn("Could not create bbolt cache directory, disk caching disabled.", "path", dbDir, "error", err)
		}
	} else {
		logger.Warn("Could not determine user cache directory, disk caching disabled.", "error", err)
	}

	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second}
		db, err = bbolt.Open(dbPath, 0600, opts)
		if err != nil {
			logger.Warn("Failed to open bbolt cache file, disk caching disabled.", "path", dbPath, "error", err)
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
				logger.Warn("Failed to ensure bbolt bucket exists, disk caching disabled.", "error", err)
				db.Close()
				db = nil
			} else {
				logger.Info("Using bbolt disk cache", "path", dbPath, "schema_version", cacheSchemaVersion)
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
		logger.Warn("Failed to create ristretto memory cache, in-memory caching disabled.", "error", cacheErr)
		memCache = nil
	} else {
		logger.Info("Initialized ristretto in-memory cache", "max_cost", "1GB")
	}

	return &GoPackagesAnalyzer{db: db, memoryCache: memCache}
}

// Close cleans up resources used by the analyzer.
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
// Returns the populated AstContextInfo and any fatal error encountered.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, absFilename string, version int, line, col int) (info *AstContextInfo, analysisErr error) {
	logger := stdslog.Default().With("absFile", absFilename, "version", version, "line", line, "col", col)
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
		if len(info.AnalysisErrors) > 0 && analysisErr == nil {
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr)
		} else if len(info.AnalysisErrors) > 0 && analysisErr != nil {
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", analysisErr, finalErr)
		}
	}()

	logger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)
	goModHash := calculateGoModHash(dir) // Util func
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
				return nil
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				logger.Debug("Bbolt cache miss.", "key", string(cacheKey))
				return nil
			}

			logger.Debug("Bbolt cache hit (raw bytes found). Decoding entry...", "key", string(cacheKey))
			var decoded CachedAnalysisEntry
			if err := gob.NewDecoder(bytes.NewReader(valBytes)).Decode(&decoded); err != nil {
				return fmt.Errorf("%w: %w", ErrCacheDecode, err)
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				logger.Warn("Bbolt cache data has old schema version. Ignoring.", "key", string(cacheKey), "cached_version", decoded.SchemaVersion, "expected_version", cacheSchemaVersion)
				return nil
			}
			cachedEntry = &decoded
			return nil
		})
		if dbViewErr != nil {
			logger.Warn("Error reading or decoding from bbolt cache. Cache check failed.", "error", dbViewErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbViewErr), logger)
			if errors.Is(dbViewErr, ErrCacheDecode) {
				go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "decode_failure")) // Util func
			}
			cachedEntry = nil
		}
		logger.Debug("Bbolt cache read attempt finished", "duration", time.Since(readStart))

		// --- Bbolt Cache Validation ---
		if cachedEntry != nil {
			validationStart := time.Now()
			logger.Debug("Potential bbolt cache hit. Validating file hashes...", "key", string(cacheKey))
			currentHashes, hashErr := calculateInputHashes(dir, nil)                                                                   // Util func
			if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes) { // Util func
				logger.Debug("Bbolt cache VALID. Attempting to decode analysis data...", "key", string(cacheKey))
				decodeStart := time.Now()
				var analysisData CachedAnalysisData
				if decodeErr := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob)).Decode(&analysisData); decodeErr == nil {
					info.PackageName = analysisData.PackageName
					info.PromptPreamble = analysisData.PromptPreamble
					cacheHit = true
					loadDuration = time.Since(decodeStart)
					logger.Debug("Analysis data successfully decoded from bbolt cache.", "duration", loadDuration)
					logger.Debug("Re-running load/analysis steps for diagnostics/hover despite cache hit.")
					cacheHit = false // Force re-analysis
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

	// --- Perform Full Analysis if Cache Miss or Invalid ---
	if !cacheHit {
		logger.Debug("Cache miss or invalid. Performing full analysis...", "key", string(cacheKey))
		loadStart := time.Now()
		fset := token.NewFileSet()
		info.TargetFileSet = fset

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
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg) // Util func
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
							logger.Debug("Writing bytes to bbolt cache", "key", string(cacheKey), "bytes", len(encodedBytes))
							return b.Put(cacheKey, encodedBytes)
						})
						if saveErr == nil {
							logger.Debug("Saved analysis results to bbolt cache", "key", string(cacheKey), "duration", time.Since(saveStart))
						} else {
							logger.Warn("Failed to write to bbolt cache", "key", string(cacheKey), "error", saveErr)
							addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheWrite, saveErr), logger)
						}
					} else {
						logger.Warn("Failed to gob-encode cache entry", "error", entryEncodeErr)
						addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr), logger)
					}
				} else {
					logger.Warn("Failed to gob-encode analysis data", "error", encodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr), logger)
				}
			} else {
				logger.Warn("Failed to calculate input hashes for cache save", "error", hashErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger)
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
	goModHash := calculateGoModHash(dir) // Util func
	cacheKey := []byte(dir + "::" + goModHash)
	logger.Info("Invalidating bbolt cache entry", "key", string(cacheKey))
	return deleteCacheEntryByKey(db, cacheKey, logger) // Util func
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
// Cycle 2 Fix: Moved method back here to satisfy Analyzer interface.
func (a *GoPackagesAnalyzer) MemoryCacheEnabled() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.memoryCache != nil
}

// GetMemoryCacheMetrics returns the performance metrics collected by Ristretto.
// Cycle 2 Fix: Moved method back here to satisfy Analyzer interface.
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
	config    Config          // Current active configuration
	configMu  sync.RWMutex    // Mutex to protect concurrent access to config.
	logger    *stdslog.Logger // Logger instance
}

// NewDeepCompleter creates a new DeepCompleter service instance.
func NewDeepCompleter(logger *stdslog.Logger) (*DeepCompleter, error) {
	if logger == nil {
		logger = stdslog.Default()
	}

	cfg, configErr := LoadConfig(logger)
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		logger.Error("Fatal error during initial config load", "error", configErr)
		return nil, configErr
	}
	if err := cfg.Validate(logger); err != nil {
		if errors.Is(err, ErrInvalidConfig) {
			logger.Error("Loaded/default config is invalid", "error", err)
			return nil, fmt.Errorf("initial config validation failed: %w", err)
		}
		logger.Warn("Initial config validation reported issues", "error", err)
	}

	analyzer := NewGoPackagesAnalyzer(logger)
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer, // Analyzer now correctly implements the interface
		formatter: newTemplateFormatter(),
		config:    cfg,
		logger:    logger,
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
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	if err := config.Validate(logger); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	analyzer := NewGoPackagesAnalyzer(logger)
	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer, // Analyzer now correctly implements the interface
		formatter: newTemplateFormatter(),
		config:    config,
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

// UpdateConfig atomically updates the completer's configuration.
func (dc *DeepCompleter) UpdateConfig(newConfig Config) error {
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate
	}
	if err := newConfig.Validate(dc.logger); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	dc.configMu.Lock()
	defer dc.configMu.Unlock()
	dc.config = newConfig

	dc.logger.Info("DeepCompleter configuration updated",
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
		streamErr := streamCompletion(streamCtx, reader, &buffer) // Util func
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // Util func
	if err != nil {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) {
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

// GetCompletionStreamFromFile provides context-aware code completion by analyzing the
// specified file and position, then streams the LLM response to the provided writer.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, absFilename string, version int, line, col int, w io.Writer) error {
	logger := dc.logger.With("operation", "GetCompletionStreamFromFile", "path", absFilename, "version", version, "line", line, "col", col)
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

	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col) // Util func
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
		streamErr := streamCompletion(apiCtx, reader, w) // Util func
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		return nil
	}

	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // Util func
	if err != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) {
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
	} else {
		logger.Info("Completion stream successful")
	}
	return nil
}
