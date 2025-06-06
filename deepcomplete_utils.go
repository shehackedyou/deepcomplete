// deepcomplete/deepcomplete_utils.go
// Contains utility functions used across the deepcomplete package.
package deepcomplete

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"go/token"
	"io"
	"log/slog"
	"math/rand" // For jitter
	"net/http"  // Needed for OllamaError status codes
	"net/url"
	"os"
	"path/filepath"
	"regexp" // Added for cleaning output
	"strings"
	"sync"
	"time"
	"unicode/utf8" // For UTF-16 conversion

	"go.etcd.io/bbolt"
	"golang.org/x/tools/go/packages"
)

// ============================================================================
// Terminal Colors (Optional - For CLI)
// ============================================================================
var (
	ColorReset  = "\033[0m"
	ColorGreen  = "\033[38;5;119m"
	ColorYellow = "\033[38;5;220m"
	ColorBlue   = "\033[38;5;153m"
	ColorRed    = "\033[38;5;203m"
	ColorCyan   = "\033[38;5;141m"
)

// PrettyPrint prints colored text to stderr.
func PrettyPrint(color, text string) {
	fmt.Fprint(os.Stderr, color, text, ColorReset)
}

// ============================================================================
// Logging Helpers
// ============================================================================

// ParseLogLevel converts a log level string to its corresponding slog.Level constant.
func ParseLogLevel(levelStr string) (slog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(levelStr)) {
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

// ============================================================================
// Configuration Loading Helpers
// ============================================================================

// GetConfigPaths determines the primary (XDG_CONFIG_HOME) and secondary (~/.config) config paths.
func GetConfigPaths(logger *slog.Logger) (primary string, secondary string, err error) {
	if logger == nil {
		logger = slog.Default()
	}
	var cfgErr, homeErr error

	userConfigDir, cfgErr := os.UserConfigDir()
	if cfgErr == nil {
		primary = filepath.Join(userConfigDir, configDirName, defaultConfigFileName)
	} else {
		logger.Warn("Could not determine user config directory (XDG)", "error", cfgErr)
	}

	homeDir, homeErr := os.UserHomeDir()
	if homeErr == nil {
		secondaryCandidate := filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
		if primary == "" && cfgErr != nil {
			primary = secondaryCandidate
			logger.Debug("Using fallback primary config path", "path", primary)
		} else if primary != secondaryCandidate {
			secondary = secondaryCandidate
		}
	} else {
		logger.Warn("Could not determine user home directory", "error", homeErr)
	}

	if primary == "" {
		err = fmt.Errorf("cannot determine config/home directories: config error: %v; home error: %v", cfgErr, homeErr)
	}
	return primary, secondary, err
}

// LoadAndMergeConfig attempts to load config from a specific path and merge its
// fields into the provided cfg object.
func LoadAndMergeConfig(path string, cfg *Config, logger *slog.Logger) (loaded bool, err error) {
	if logger == nil {
		logger = slog.Default()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("reading config file %q failed: %w", path, err)
	}
	loaded = true

	if len(data) == 0 {
		logger.Warn("Config file exists but is empty, ignoring.", "path", path)
		return loaded, nil
	}

	var fileCfg FileConfig
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&fileCfg); err != nil {
		return loaded, fmt.Errorf("parsing config file JSON %q failed: %w", path, err)
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
	// Merge MemoryCacheTTLSeconds if present
	if fileCfg.MemoryCacheTTLSeconds != nil {
		cfg.MemoryCacheTTLSeconds = *fileCfg.MemoryCacheTTLSeconds
		// Update the derived duration field as well
		cfg.MemoryCacheTTL = time.Duration(cfg.MemoryCacheTTLSeconds) * time.Second
		mergedFields++
	}

	logger.Debug("Merged configuration from file", "path", path, "fields_merged", mergedFields)
	return loaded, nil
}

// WriteDefaultConfig creates the config directory if needed and writes the
// default configuration values as a JSON file.
func WriteDefaultConfig(path string, defaultConfig Config, logger *slog.Logger) error {
	if logger == nil {
		logger = slog.Default()
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0750); err != nil {
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}

	// Use FileConfig structure for writing to ensure only configurable fields are included
	// Ensure pointers are correctly assigned
	stopCopy := make([]string, len(defaultConfig.Stop))
	copy(stopCopy, defaultConfig.Stop)

	expCfg := FileConfig{
		OllamaURL:             &defaultConfig.OllamaURL,
		Model:                 &defaultConfig.Model,
		MaxTokens:             &defaultConfig.MaxTokens,
		Stop:                  &stopCopy, // Assign pointer to the copy
		Temperature:           &defaultConfig.Temperature,
		LogLevel:              &defaultConfig.LogLevel,
		UseAst:                &defaultConfig.UseAst,
		UseFim:                &defaultConfig.UseFim,
		MaxPreambleLen:        &defaultConfig.MaxPreambleLen,
		MaxSnippetLen:         &defaultConfig.MaxSnippetLen,
		MemoryCacheTTLSeconds: &defaultConfig.MemoryCacheTTLSeconds, // Add TTL field
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

// ============================================================================
// LSP Position Conversion Helpers
// ============================================================================

// LspPositionToBytePosition converts 0-based LSP line/character (UTF-16) to
// 1-based Go line/column (bytes) and 0-based byte offset.
func LspPositionToBytePosition(content []byte, lspPos LSPPosition, logger *slog.Logger) (line, col, byteOffset int, err error) {
	if logger == nil {
		logger = slog.Default()
	}

	if content == nil {
		return 0, 0, -1, fmt.Errorf("%w: file content is nil", ErrPositionConversion)
	}
	targetLine := int(int32(lspPos.Line))
	targetUTF16Char := int(int32(lspPos.Character))

	if targetLine < 0 {
		return 0, 0, -1, fmt.Errorf("%w: line number %d must be >= 0", ErrInvalidPositionInput, targetLine)
	}
	if targetUTF16Char < 0 {
		return 0, 0, -1, fmt.Errorf("%w: character offset %d must be >= 0", ErrInvalidPositionInput, targetUTF16Char)
	}

	currentLine := 0
	currentByteOffset := 0
	scanner := bufio.NewScanner(bytes.NewReader(content))
	for scanner.Scan() {
		lineTextBytes := scanner.Bytes()
		lineLengthBytes := len(lineTextBytes)
		newlineLengthBytes := 1 // Assume \n

		if currentLine == targetLine {
			byteOffsetInLine, convErr := Utf16OffsetToBytes(lineTextBytes, targetUTF16Char, logger) // Pass logger
			if convErr != nil {
				if errors.Is(convErr, ErrPositionOutOfRange) {
					logger.Warn("UTF16 offset out of range, clamping to line end",
						"line", targetLine,
						"char", targetUTF16Char,
						"line_length_bytes", lineLengthBytes,
						"error", convErr)
					byteOffsetInLine = lineLengthBytes
				} else {
					return 0, 0, -1, fmt.Errorf("failed converting UTF16 to byte offset on line %d: %w", currentLine, convErr)
				}
			}
			line = currentLine + 1
			col = byteOffsetInLine + 1
			byteOffset = currentByteOffset + byteOffsetInLine
			return line, col, byteOffset, nil
		}
		currentByteOffset += lineLengthBytes + newlineLengthBytes
		currentLine++
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, -1, fmt.Errorf("%w: error scanning file content: %w", ErrPositionConversion, err)
	}

	if currentLine == targetLine {
		if targetUTF16Char == 0 {
			line = currentLine + 1
			col = 1
			byteOffset = currentByteOffset
			return line, col, byteOffset, nil
		}
		return 0, 0, -1, fmt.Errorf("%w: invalid character offset %d on line %d (after last line with content)", ErrPositionOutOfRange, targetUTF16Char, targetLine)
	}

	return 0, 0, -1, fmt.Errorf("%w: LSP line %d not found in file (total lines scanned %d)", ErrPositionOutOfRange, targetLine, currentLine)
}

// Utf16OffsetToBytes converts a 0-based UTF-16 offset within a line to a 0-based byte offset.
func Utf16OffsetToBytes(line []byte, utf16Offset int, logger *slog.Logger) (int, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if utf16Offset < 0 {
		return 0, fmt.Errorf("%w: invalid utf16Offset: %d (must be >= 0)", ErrInvalidPositionInput, utf16Offset)
	}
	if utf16Offset == 0 {
		return 0, nil
	}

	byteOffset := 0
	currentUTF16Offset := 0
	lineLenBytes := len(line)

	for byteOffset < lineLenBytes {
		if currentUTF16Offset >= utf16Offset {
			break
		}
		r, size := utf8.DecodeRune(line[byteOffset:])
		if r == utf8.RuneError && size <= 1 {
			return byteOffset, fmt.Errorf("%w at byte offset %d", ErrInvalidUTF8, byteOffset)
		}

		utf16Units := 1
		if r > 0xFFFF {
			utf16Units = 2
		}

		if currentUTF16Offset+utf16Units > utf16Offset {
			break
		}

		currentUTF16Offset += utf16Units
		byteOffset += size

		if currentUTF16Offset == utf16Offset {
			break
		}
	}

	if currentUTF16Offset < utf16Offset {
		return lineLenBytes, fmt.Errorf("%w: utf16Offset %d is beyond the line length in UTF-16 units (%d)", ErrPositionOutOfRange, utf16Offset, currentUTF16Offset)
	}

	return byteOffset, nil
}

// byteOffsetToLSPPosition converts a 0-based byte offset to 0-based LSP line/char (UTF-16).
func byteOffsetToLSPPosition(content []byte, targetByteOffset int, logger *slog.Logger) (line, char uint32, err error) {
	if logger == nil {
		logger = slog.Default()
	}
	if content == nil {
		return 0, 0, errors.New("content is nil")
	}
	if targetByteOffset < 0 {
		return 0, 0, fmt.Errorf("%w: invalid targetByteOffset: %d", ErrInvalidPositionInput, targetByteOffset)
	}
	if targetByteOffset > len(content) {
		logger.Debug("targetByteOffset exceeds content length, clamping to EOF", "offset", targetByteOffset, "content_len", len(content))
		targetByteOffset = len(content)
	}

	currentLine := uint32(0)
	currentByteOffset := 0
	currentLineStartByteOffset := 0

	for currentByteOffset < targetByteOffset {
		r, size := utf8.DecodeRune(content[currentByteOffset:])
		if r == utf8.RuneError && size <= 1 {
			return 0, 0, fmt.Errorf("%w at byte offset %d", ErrInvalidUTF8, currentByteOffset)
		}
		if r == '\n' {
			currentLine++
			currentLineStartByteOffset = currentByteOffset + size
		}
		currentByteOffset += size
	}

	lineContentBytes := content[currentLineStartByteOffset:targetByteOffset]
	utf16CharOffset, convErr := bytesToUTF16Offset(lineContentBytes, logger) // Pass logger
	if convErr != nil {
		logger.Error("Error converting line bytes to UTF16 offset", "error", convErr, "line", currentLine)
		utf16CharOffset = len(lineContentBytes) // Fallback
	}

	return currentLine, uint32(utf16CharOffset), nil
}

// bytesToUTF16Offset calculates the number of UTF-16 code units for a byte slice.
func bytesToUTF16Offset(bytes []byte, logger *slog.Logger) (int, error) {
	if logger == nil {
		logger = slog.Default()
	}
	utf16Offset := 0
	byteOffset := 0
	for byteOffset < len(bytes) {
		r, size := utf8.DecodeRune(bytes[byteOffset:])
		if r == utf8.RuneError && size <= 1 {
			return utf16Offset, fmt.Errorf("%w at byte offset %d within slice", ErrInvalidUTF8, byteOffset)
		}
		if r > 0xFFFF {
			utf16Offset += 2
		} else {
			utf16Offset += 1
		}
		byteOffset += size
	}
	return utf16Offset, nil
}

// ============================================================================
// Path Validation & URI Conversion Helpers
// ============================================================================

// ValidateAndGetFilePath converts a file:// DocumentURI string to a clean, absolute local path.
func ValidateAndGetFilePath(uri string, logger *slog.Logger) (string, error) {
	if logger == nil {
		logger = slog.Default()
	}

	if uri == "" {
		logger.Warn("ValidateAndGetFilePath called with empty URI")
		return "", errors.New("document URI cannot be empty")
	}
	logger.Debug("Validating URI", "uri", uri)

	parsedURL, err := url.Parse(uri)
	if err != nil {
		logger.Warn("Failed to parse document URI", "uri", uri, "error", err)
		return "", fmt.Errorf("%w: invalid document URI '%s': %w", ErrInvalidURI, uri, err)
	}

	if parsedURL.Scheme != "file" {
		logger.Warn("Received non-file document URI", "uri", uri, "scheme", parsedURL.Scheme)
		return "", fmt.Errorf("%w: unsupported URI scheme: '%s' (only 'file://' is supported)", ErrInvalidURI, parsedURL.Scheme)
	}

	filePath := parsedURL.Path
	if parsedURL.Host != "" && strings.ToLower(parsedURL.Host) != "localhost" {
		logger.Warn("File URI includes unexpected host component", "uri", uri, "host", parsedURL.Host)
		return "", fmt.Errorf("%w: file URI should not contain host component for local files (host: %s)", ErrInvalidURI, parsedURL.Host)
	}

	absPath, err := filepath.Abs(filePath)
	if err != nil {
		logger.Warn("Failed to get absolute path", "uri_path", filePath, "error", err)
		return "", fmt.Errorf("failed to resolve absolute path for '%s': %w", filePath, err)
	}

	logger.Debug("Validated and converted URI to path", "uri", uri, "path", absPath)
	return absPath, nil
}

// PathToURI converts an absolute file path to a file:// URI.
func PathToURI(absPath string) (string, error) {
	if !filepath.IsAbs(absPath) {
		return "", fmt.Errorf("path must be absolute: %s", absPath)
	}
	absPath = filepath.ToSlash(absPath) // Ensure forward slashes for URI

	uri := url.URL{
		Scheme: "file",
		Path:   absPath,
	}
	return uri.String(), nil
}

// ============================================================================
// Cache Helper Functions
// ============================================================================

// calculateGoModHash calculates the SHA256 hash of the go.mod file.
func calculateGoModHash(dir string, logger *slog.Logger) string {
	if logger == nil {
		logger = slog.Default()
	}
	logger = logger.With("dir", dir)

	goModPath := filepath.Join(dir, "go.mod")
	f, err := os.Open(goModPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			logger.Debug("go.mod not found, using 'no-gomod' hash")
			return "no-gomod"
		}
		logger.Warn("Error reading go.mod for hashing", "path", goModPath, "error", err)
		return "read-error"
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		logger.Warn("Error hashing go.mod content", "path", goModPath, "error", err)
		return "hash-error"
	}
	return hex.EncodeToString(h.Sum(nil))
}

// calculateInputHashes calculates hashes for go.mod, go.sum, and Go files.
func calculateInputHashes(dir string, pkg *packages.Package, logger *slog.Logger) (map[string]string, error) {
	if logger == nil {
		logger = slog.Default()
	}
	hashes := make(map[string]string)
	filesToHash := make(map[string]struct{})
	logger = logger.With("dir", dir)

	for _, fname := range []string{"go.mod", "go.sum"} {
		fpath := filepath.Join(dir, fname)
		if _, err := os.Stat(fpath); err == nil {
			if absPath, absErr := filepath.Abs(fpath); absErr == nil {
				filesToHash[absPath] = struct{}{}
			} else {
				logger.Warn("Could not get absolute path for hashing", "file", fpath, "error", absErr)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			logger.Warn("Failed to stat file for hashing", "file", fpath, "error", err)
		}
	}

	filesFromPkg := false
	if pkg != nil && len(pkg.CompiledGoFiles) > 0 {
		if len(pkg.CompiledGoFiles[0]) > 0 {
			firstFileAbs, _ := filepath.Abs(pkg.CompiledGoFiles[0])
			dirAbs, _ := filepath.Abs(dir)
			if filepath.IsAbs(firstFileAbs) && strings.HasPrefix(firstFileAbs, dirAbs) {
				filesFromPkg = true
				logger.Debug("Hashing based on CompiledGoFiles", "count", len(pkg.CompiledGoFiles), "package", pkg.ID)
				for _, fpath := range pkg.CompiledGoFiles {
					if absPath, absErr := filepath.Abs(fpath); absErr == nil {
						filesToHash[absPath] = struct{}{}
					} else {
						logger.Warn("Could not get absolute path for compiled file, skipping hash", "file", fpath, "error", absErr)
					}
				}
			} else {
				logger.Warn("CompiledGoFiles paths seem invalid or not absolute/relative to dir, falling back to directory scan.",
					"first_file", pkg.CompiledGoFiles[0], "dir", dir)
			}
		} else {
			logger.Warn("First entry in CompiledGoFiles is empty, falling back to directory scan.")
		}
	}

	if !filesFromPkg {
		logger.Debug("Calculating input hashes by scanning directory")
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, fmt.Errorf("failed to scan directory %s for hashing: %w", dir, err)
		}
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") && !strings.HasSuffix(entry.Name(), "_test.go") {
				absPath := filepath.Join(dir, entry.Name())
				filesToHash[absPath] = struct{}{}
			}
		}
	}

	for absPath := range filesToHash {
		relPath, err := filepath.Rel(dir, absPath)
		if err != nil {
			logger.Warn("Could not get relative path for hashing, using base name", "absPath", absPath, "error", err)
			relPath = filepath.Base(absPath)
		}
		relPath = filepath.ToSlash(relPath)

		hash, err := hashFileContent(absPath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				logger.Warn("File disappeared during hashing, skipping.", "path", absPath)
				continue
			}
			return nil, fmt.Errorf("%w: failed to hash input file %s: %w", ErrCacheHash, absPath, err)
		}
		hashes[relPath] = hash
	}
	logger.Debug("Calculated hashes for input files", "count", len(hashes))
	return hashes, nil
}

// hashFileContent calculates the SHA256 hash of a single file.
func hashFileContent(filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// compareFileHashes compares current and cached file hashes, logging differences.
func compareFileHashes(current, cached map[string]string, logger *slog.Logger) bool {
	if logger == nil {
		logger = slog.Default()
	}

	if len(current) != len(cached) {
		logger.Debug("Cache invalid: File count mismatch", "current_count", len(current), "cached_count", len(cached))
		if logger.Enabled(context.Background(), slog.LevelDebug) {
			for relPath := range current {
				if _, ok := cached[relPath]; !ok {
					logger.Debug("File added since cache", "file", relPath)
				}
			}
			for relPath := range cached {
				if _, ok := current[relPath]; !ok {
					logger.Debug("File removed since cache", "file", relPath)
				}
			}
		}
		return false
	}
	for relPath, currentHash := range current {
		cachedHash, ok := cached[relPath]
		if !ok {
			logger.Debug("Cache invalid: File missing in cache despite matching counts", "file", relPath)
			return false
		}
		if currentHash != cachedHash {
			logger.Debug("Cache invalid: Hash mismatch", "file", relPath)
			return false
		}
	}
	return true
}

// deleteCacheEntryByKey removes an entry directly using the key.
func deleteCacheEntryByKey(db *bbolt.DB, cacheKey []byte, logger *slog.Logger) error {
	if db == nil {
		return errors.New("cannot delete cache entry: db is nil")
	}
	if logger == nil {
		logger = slog.Default()
	}
	logger = logger.With("cache_key", string(cacheKey))

	err := db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(cacheBucketName)
		if b == nil {
			logger.Warn("Cache bucket not found during delete attempt.")
			return nil
		}
		if b.Get(cacheKey) == nil {
			logger.Debug("Cache key not found during delete attempt, nothing to delete.")
			return nil
		}
		logger.Debug("Deleting cache entry")
		return b.Delete(cacheKey)
	})
	if err != nil {
		logger.Warn("Failed to delete cache entry", "error", err)
		return fmt.Errorf("%w: failed to delete entry %s: %w", ErrCacheWrite, string(cacheKey), err)
	}
	return nil
}

// ============================================================================
// Retry Helper
// ============================================================================

// retry executes an operation function with backoff and retry logic.
// It respects the context cancellation and applies exponential backoff with jitter.
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration, logger *slog.Logger) error {
	var lastErr error
	if logger == nil {
		logger = slog.Default()
	}

	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		attemptLogger := logger.With("attempt", i+1, "max_attempts", maxRetries)

		// Check context before executing the operation
		select {
		case <-ctx.Done():
			attemptLogger.Warn("Context cancelled before attempt", "error", ctx.Err())
			return fmt.Errorf("retry cancelled before attempt %d: %w", i+1, ctx.Err())
		default:
		}

		lastErr = operation()
		if lastErr == nil {
			return nil // Success
		}

		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			attemptLogger.Warn("Attempt failed due to context error. Not retrying.", "error", lastErr)
			return lastErr
		}

		var ollamaErr *OllamaError
		isRetryableOllama := errors.As(lastErr, &ollamaErr) &&
			(ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests || ollamaErr.Status == http.StatusInternalServerError)
		isRetryableNetwork := errors.Is(lastErr, ErrOllamaUnavailable) || errors.Is(lastErr, ErrStreamProcessing)
		isRetryable := isRetryableOllama || isRetryableNetwork

		if !isRetryable {
			attemptLogger.Warn("Attempt failed with non-retryable error.", "error", lastErr)
			return lastErr
		}

		if i == maxRetries-1 {
			break
		}

		jitter := time.Duration(rand.Int63n(int64(currentDelay) / 4))
		waitDuration := currentDelay + jitter
		attemptLogger.Warn("Attempt failed with retryable error. Retrying...", "error", lastErr, "delay", waitDuration)

		select {
		case <-ctx.Done():
			attemptLogger.Warn("Context cancelled during retry wait", "error", ctx.Err())
			return fmt.Errorf("retry cancelled during wait after attempt %d: %w", i+1, ctx.Err())
		case <-time.After(waitDuration):
			currentDelay *= 2
			maxDelay := 5 * time.Second
			if currentDelay > maxDelay {
				currentDelay = maxDelay
			}
		}
	}
	logger.Error("Operation failed after all retries.", "retries", maxRetries, "final_error", lastErr)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr)
}

// ============================================================================
// Spinner (CLI Helper)
// ============================================================================
type Spinner struct {
	chars    []string
	message  string
	index    int
	mu       sync.Mutex
	stopChan chan struct{}
	doneChan chan struct{}
	running  bool
	logger   *slog.Logger
}

func NewSpinner(logger *slog.Logger) *Spinner {
	if logger == nil {
		logger = slog.Default()
	}
	return &Spinner{
		chars:  []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
		index:  0,
		logger: logger.With("component", "Spinner"),
	}
}

func (s *Spinner) Start(initialMessage string) {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.stopChan = make(chan struct{})
	s.doneChan = make(chan struct{})
	s.message = initialMessage
	s.running = true
	s.mu.Unlock()

	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		defer func() {
			s.mu.Lock()
			isRunning := s.running
			s.running = false
			s.mu.Unlock()
			if isRunning {
				fmt.Fprintf(os.Stderr, "\r\033[K")
			}
			close(s.doneChan)
		}()
		for {
			select {
			case <-s.stopChan:
				return
			case <-ticker.C:
				s.mu.Lock()
				if !s.running {
					s.mu.Unlock()
					return
				}
				char := s.chars[s.index]
				msg := s.message
				s.index = (s.index + 1) % len(s.chars)
				s.mu.Unlock()
				fmt.Fprintf(os.Stderr, "\r\033[K%s%s%s %s", ColorCyan, char, ColorReset, msg)
			}
		}
	}()
}

func (s *Spinner) UpdateMessage(newMessage string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		s.message = newMessage
	}
}

func (s *Spinner) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	select {
	case <-s.stopChan:
	default:
		close(s.stopChan)
	}
	doneChan := s.doneChan
	s.mu.Unlock()

	if doneChan != nil {
		select {
		case <-doneChan:
		case <-time.After(500 * time.Millisecond):
			s.logger.Warn("Timeout waiting for spinner goroutine cleanup")
		}
	}
	fmt.Fprintf(os.Stderr, "\r\033[K")
}

// ============================================================================
// Snippet Extraction Helper
// ============================================================================

type SnippetContext struct {
	Prefix          string
	Suffix          string
	FullLine        string
	FallbackContext string // Added field for fallback context when AST is disabled
}

// extractSnippetContext extracts code prefix, suffix, and full line around the cursor.
// If useAST is false, it attempts to include the enclosing function body as fallback context.
func extractSnippetContext(filename string, row, col int, useAST bool, logger *slog.Logger) (SnippetContext, error) {
	if logger == nil {
		logger = slog.Default()
	}
	logger = logger.With("file", filename, "line", row, "col", col, "useAST", useAST)
	var ctx SnippetContext

	contentBytes, err := os.ReadFile(filename)
	if err != nil {
		return ctx, fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	content := string(contentBytes)

	fset := token.NewFileSet()
	file := fset.AddFile(filename, 1, len(contentBytes))
	if file == nil {
		return ctx, fmt.Errorf("failed to add file '%s' to fileset", filename)
	}

	cursorPos, posErr := calculateCursorPos(file, row, col, logger) // Pass logger
	if posErr != nil {
		return ctx, fmt.Errorf("cannot determine valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		return ctx, fmt.Errorf("%w: invalid cursor position calculated (Pos: %d)", ErrPositionConversion, cursorPos)
	}

	offset := file.Offset(cursorPos)
	if offset < 0 {
		logger.Warn("Calculated negative offset from position, clamping to 0", "pos", cursorPos, "offset", offset)
		offset = 0
	}
	if offset > len(content) {
		logger.Warn("Calculated offset beyond content length, clamping to EOF", "pos", cursorPos, "offset", offset, "content_len", len(content))
		offset = len(content)
	}

	ctx.Prefix = content[:offset]
	ctx.Suffix = content[offset:]

	// --- Extract Full Line ---
	lineStartPos := file.LineStart(row)
	if !lineStartPos.IsValid() {
		logger.Warn("Could not get start position for line, cannot extract full line", "line", row)
	} else {
		startOffset := file.Offset(lineStartPos)
		lineEndOffset := file.Size()
		fileLineCount := file.LineCount()
		if row < fileLineCount {
			nextLineStartPos := file.LineStart(row + 1)
			if nextLineStartPos.IsValid() {
				lineEndOffset = file.Offset(nextLineStartPos)
			} else {
				logger.Warn("Could not get start position for next line", "line", row+1)
			}
		}

		if startOffset >= 0 && lineEndOffset >= startOffset && lineEndOffset <= len(content) {
			lineContent := content[startOffset:lineEndOffset]
			lineContent = strings.TrimRight(lineContent, "\n")
			lineContent = strings.TrimRight(lineContent, "\r")
			ctx.FullLine = lineContent
		} else {
			logger.Warn("Could not extract full line content due to invalid offsets",
				"startOffset", startOffset,
				"lineEndOffset", lineEndOffset,
				"contentLen", len(content))
		}
	}

	// --- Fallback Context if AST is Disabled ---
	if !useAST {
		logger.Debug("AST disabled, attempting to extract fallback context (enclosing function)")
		body, found := findEnclosingFuncBody(contentBytes, offset)
		if found {
			ctx.FallbackContext = "// Fallback Context (Enclosing Function Body - Approx.):\n" + body
			logger.Debug("Extracted fallback function body context", "length", len(body))
		} else {
			logger.Debug("Could not find enclosing function body for fallback context")
		}
	}

	return ctx, nil
}

// calculateCursorPos converts 1-based line/col to 0-based token.Pos offset.
func calculateCursorPos(file *token.File, line, col int, logger *slog.Logger) (token.Pos, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if line <= 0 {
		return token.NoPos, fmt.Errorf("%w: line number %d must be >= 1", ErrInvalidPositionInput, line)
	}
	if col <= 0 {
		return token.NoPos, fmt.Errorf("%w: column number %d must be >= 1", ErrInvalidPositionInput, col)
	}
	if file == nil {
		return token.NoPos, errors.New("invalid token.File (nil)")
	}

	fileLineCount := file.LineCount()
	if line > fileLineCount+1 || (line == fileLineCount+1 && col > 1) {
		return token.NoPos, fmt.Errorf("%w: line number %d / column %d exceeds file bounds (%d lines)", ErrPositionOutOfRange, line, col, fileLineCount)
	}

	if line == fileLineCount+1 {
		if col == 1 {
			return file.Pos(file.Size()), nil
		}
		return token.NoPos, fmt.Errorf("%w: column %d invalid on virtual line %d (only col 1 allowed)", ErrPositionOutOfRange, col, line)
	}

	lineStartPos := file.LineStart(line)
	if !lineStartPos.IsValid() {
		return token.NoPos, fmt.Errorf("%w: cannot get start offset for line %d in file '%s'", ErrPositionConversion, line, file.Name())
	}

	lineStartOffset := file.Offset(lineStartPos)
	targetOffset := lineStartOffset + col - 1

	lineEndOffset := file.Size()
	if line < fileLineCount {
		nextLineStartPos := file.LineStart(line + 1)
		if nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	finalOffset := targetOffset
	clamped := false
	if targetOffset < lineStartOffset {
		finalOffset = lineStartOffset
		clamped = true
		logger.Warn("Column resulted in offset before line start. Clamping.", "line", line, "col", col, "targetOffset", targetOffset, "lineStartOffset", lineStartOffset)
	}
	if targetOffset > lineEndOffset {
		finalOffset = lineEndOffset
		clamped = true
		logger.Warn("Column resulted in offset beyond line end. Clamping.", "line", line, "col", col, "targetOffset", targetOffset, "lineEndOffset", lineEndOffset)
	}

	pos := file.Pos(finalOffset)
	if !pos.IsValid() {
		logger.Error("Clamped offset resulted in invalid token.Pos. Falling back to line start.", "offset", finalOffset, "line_start_pos", lineStartPos)
		return lineStartPos, fmt.Errorf("%w: failed to calculate valid token.Pos for offset %d (clamped: %v)", ErrPositionConversion, finalOffset, clamped)
	}
	return pos, nil
}

// ============================================================================
// Stream Processing Helpers
// ============================================================================

// Precompile regex for cleaning markdown code fences and FIM tokens
var mdCodeFenceRegex = regexp.MustCompile("(?m)^```(?:go)?\n|```$")
var fimTokenRegex = regexp.MustCompile(fmt.Sprintf("%s|%s|%s|%s", regexp.QuoteMeta(FimPrefixToken), regexp.QuoteMeta(FimMiddleToken), regexp.QuoteMeta(FimSuffixToken), regexp.QuoteMeta(FimEOTToken)))

// cleanLLMOutput removes FIM tokens and markdown code fences.
func cleanLLMOutput(rawOutput string) string {
	// Remove markdown code fences first
	cleaned := mdCodeFenceRegex.ReplaceAllString(rawOutput, "")
	// Then remove FIM tokens
	cleaned = fimTokenRegex.ReplaceAllString(cleaned, "")
	// Trim leading/trailing whitespace that might remain
	return strings.TrimSpace(cleaned)
}

// streamCompletion reads the Ollama stream response, cleans it, and writes it to w.
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer, logger *slog.Logger) error {
	defer r.Close()
	reader := bufio.NewReader(r)
	lineCount := 0
	if logger == nil {
		logger = slog.Default()
	}

	for {
		select {
		case <-ctx.Done():
			logger.Warn("Context cancelled during streaming", "error", ctx.Err())
			return ctx.Err() // Return context error directly
		default:
		}

		line, err := reader.ReadBytes('\n')

		if len(line) > 0 {
			// Process and clean the line before writing
			if procErr := processLine(line, w, logger); procErr != nil { // Pass logger
				return fmt.Errorf("processing stream line failed: %w", procErr)
			}
			lineCount++
		}

		if err != nil {
			if err == io.EOF {
				logger.Debug("Stream processing finished (EOF)", "lines_processed", lineCount)
				return nil
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			}
		}
	}
}

// processLine decodes a single line from the Ollama stream, cleans it, and writes the content.
func processLine(line []byte, w io.Writer, logger *slog.Logger) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil
	}
	if logger == nil {
		logger = slog.Default()
	}

	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		logger.Debug("Ignoring non-JSON line from Ollama stream", "line", string(line))
		return nil
	}

	if resp.Error != "" {
		logger.Error("Ollama stream reported an error", "error", resp.Error)
		return fmt.Errorf("%w: %s", ErrStreamProcessing, resp.Error)
	}

	// Clean the response chunk before writing
	cleanedResponse := cleanLLMOutput(resp.Response)

	if len(cleanedResponse) > 0 {
		if _, err := fmt.Fprint(w, cleanedResponse); err != nil {
			logger.Error("Error writing cleaned stream chunk to output", "error", err)
			return fmt.Errorf("error writing to output: %w", err)
		}
	}
	return nil
}

// ============================================================================
// Misc String Helpers
// ============================================================================

// firstN truncates a string to the first N runes and adds "...".
func firstN(s string, n int) string {
	if n < 0 {
		n = 0
	}
	count := 0
	for i := range s {
		if count == n {
			return s[:i] + "..."
		}
		count++
	}
	return s
}

// getPosString safely gets a position string or returns a placeholder.
func getPosString(fset *token.FileSet, pos token.Pos) string {
	if fset != nil && pos.IsValid() {
		return fset.Position(pos).String()
	}
	return fmt.Sprintf("Pos(%d)", pos)
}

// --- Helper for Fallback Context Extraction ---

// findEnclosingFuncBody attempts to find the body of the function enclosing the offset.
// This uses basic string searching and brace balancing, not full parsing.
func findEnclosingFuncBody(content []byte, offset int) (string, bool) {
	if offset < 0 || offset > len(content) {
		return "", false
	}

	searchStart := offset
	if searchStart > len(content) {
		searchStart = len(content)
	}
	prefix := content[:searchStart]
	funcKeywordIndex := strings.LastIndex(string(prefix), "\nfunc ")
	if funcKeywordIndex == -1 {
		if strings.HasPrefix(string(prefix), "func ") {
			funcKeywordIndex = 0
		} else {
			return "", false
		}
	} else {
		funcKeywordIndex++
	}

	braceSearchStart := funcKeywordIndex + len("func ")
	if braceSearchStart >= len(content) {
		return "", false
	}
	braceIndex := strings.Index(string(content[braceSearchStart:]), "{")
	if braceIndex == -1 {
		return "", false
	}
	funcBodyStartOffset := braceSearchStart + braceIndex + 1

	balance := 1
	funcBodyEndOffset := -1
	for i := funcBodyStartOffset; i < len(content); i++ {
		switch content[i] {
		case '{':
			balance++
		case '}':
			balance--
			if balance == 0 {
				funcBodyEndOffset = i
				goto foundEnd
			}
		}
	}
foundEnd:

	if funcBodyEndOffset != -1 && funcBodyEndOffset >= funcBodyStartOffset {
		// Ensure the cursor offset is actually within this found function body
		if offset >= funcBodyStartOffset && offset <= funcBodyEndOffset {
			return string(content[funcBodyStartOffset:funcBodyEndOffset]), true
		}
	}

	return "", false
}
