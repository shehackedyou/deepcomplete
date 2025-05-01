// deepcomplete_utils.go
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
	"net/http" // Needed for OllamaError status codes
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8" // For UTF-16 conversion

	"go.etcd.io/bbolt"
	"golang.org/x/tools/go/packages"
)

// ============================================================================
// Terminal Colors (Optional - Keep if CLI still uses them)
// ============================================================================
var (
	ColorReset  = "\033[0m"
	ColorGreen  = "\033[38;5;119m"
	ColorYellow = "\033[38;5;220m"
	ColorBlue   = "\033[38;5;153m"
	ColorRed    = "\033[38;5;203m"
	ColorCyan   = "\033[38;5;141m"
)

// ============================================================================
// Exported Helper Functions (If any remain needed by external callers)
// ============================================================================

// PrettyPrint prints colored text to stderr. (Consider removing if CLI uses slog)
func PrettyPrint(color, text string) {
	// Direct printing to stderr might bypass structured logging.
	// Consider replacing CLI usage with slog or keeping this specific for CLI visual feedback.
	fmt.Fprint(os.Stderr, color, text, ColorReset)
}

// ============================================================================
// LSP Position Conversion Helpers (Updated with slog)
// ============================================================================

// LspPositionToBytePosition converts 0-based LSP line/character (UTF-16) to
// 1-based Go line/column (bytes) and 0-based byte offset.
// Uses slog for logging warnings.
// Uses the LSPPosition type defined in lsp_protocol.go (implicitly within the package).
func LspPositionToBytePosition(content []byte, lspPos LSPPosition) (line, col, byteOffset int, err error) {
	logger := slog.Default() // Use default logger, assumes initialized by caller

	if content == nil {
		return 0, 0, -1, fmt.Errorf("%w: file content is nil", ErrPositionConversion)
	}
	// Use int32 for intermediate calculations to avoid potential overflow issues
	// when converting uint32 to int, especially if uint32 is near max value.
	targetLine := int(int32(lspPos.Line))
	targetUTF16Char := int(int32(lspPos.Character))

	// Validate input position non-negativity
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
		newlineLengthBytes := 1 // Assume \n, adjust if needed for \r\n

		if currentLine == targetLine {
			byteOffsetInLine, convErr := Utf16OffsetToBytes(lineTextBytes, targetUTF16Char) // Uses slog internally
			if convErr != nil {
				if errors.Is(convErr, ErrPositionOutOfRange) {
					// Log the clamping action
					logger.Warn("UTF16 offset out of range, clamping to line end",
						"line", targetLine,
						"char", targetUTF16Char,
						"line_length_bytes", lineLengthBytes,
						"error", convErr)
					byteOffsetInLine = lineLengthBytes // Clamp to end
				} else {
					// Other conversion errors (e.g., invalid UTF8)
					return 0, 0, -1, fmt.Errorf("failed converting UTF16 to byte offset on line %d: %w", currentLine, convErr)
				}
			}
			line = currentLine + 1                            // Convert 0-based line to 1-based
			col = byteOffsetInLine + 1                        // Convert 0-based byte offset in line to 1-based col
			byteOffset = currentByteOffset + byteOffsetInLine // Calculate 0-based byte offset from file start
			return line, col, byteOffset, nil                 // Success.
		}
		// Move to the start of the next line
		currentByteOffset += lineLengthBytes + newlineLengthBytes
		currentLine++
	}
	// Check for scanner errors
	if err := scanner.Err(); err != nil {
		return 0, 0, -1, fmt.Errorf("%w: error scanning file content: %w", ErrPositionConversion, err)
	}

	// Handle cursor potentially being on the line immediately after the last line of content.
	if currentLine == targetLine {
		if targetUTF16Char == 0 { // Only valid position is character 0
			line = currentLine + 1
			col = 1
			byteOffset = currentByteOffset // Offset at the end of the last line's content
			return line, col, byteOffset, nil
		}
		// Any character offset > 0 is invalid on this virtual line
		return 0, 0, -1, fmt.Errorf("%w: invalid character offset %d on line %d (after last line with content)", ErrPositionOutOfRange, targetUTF16Char, targetLine)
	}

	// Target line not found (requested line number > number of lines in file)
	return 0, 0, -1, fmt.Errorf("%w: LSP line %d not found in file (total lines scanned %d)", ErrPositionOutOfRange, targetLine, currentLine)
}

// Utf16OffsetToBytes converts a 0-based UTF-16 offset within a line to a 0-based byte offset.
// Uses slog for logging warnings.
func Utf16OffsetToBytes(line []byte, utf16Offset int) (int, error) {
	//logger := slog.Default() // Use default logger, assumes initialized by caller

	if utf16Offset < 0 {
		return 0, fmt.Errorf("%w: invalid utf16Offset: %d (must be >= 0)", ErrInvalidPositionInput, utf16Offset)
	}
	if utf16Offset == 0 {
		return 0, nil // 0 offset always maps to 0 bytes
	}

	byteOffset := 0
	currentUTF16Offset := 0
	lineLenBytes := len(line)

	for byteOffset < lineLenBytes {
		if currentUTF16Offset >= utf16Offset {
			break // Reached or passed target UTF-16 offset
		}
		r, size := utf8.DecodeRune(line[byteOffset:])
		if r == utf8.RuneError && size <= 1 {
			// Invalid UTF-8 sequence encountered
			return byteOffset, fmt.Errorf("%w at byte offset %d", ErrInvalidUTF8, byteOffset)
		}

		utf16Units := 1
		if r > 0xFFFF { // Check if rune needs surrogate pair in UTF-16
			utf16Units = 2
		}

		// Check if adding this rune *would* exceed the target offset
		if currentUTF16Offset+utf16Units > utf16Offset {
			// The target offset falls *within* the current multi-unit rune.
			// Per LSP spec, position is typically before the character,
			// so we don't advance byteOffset past the start of this rune.
			break
		}

		// Advance offsets
		currentUTF16Offset += utf16Units
		byteOffset += size

		// Check if we landed exactly on the target offset after advancing
		if currentUTF16Offset == utf16Offset {
			break
		}
	}

	// After loop, check if the requested offset was actually reachable
	if currentUTF16Offset < utf16Offset {
		// Target offset is beyond the actual UTF-16 length of the line
		// Return the byte offset corresponding to the end of the line and an error
		return lineLenBytes, fmt.Errorf("%w: utf16Offset %d is beyond the line length in UTF-16 units (%d)", ErrPositionOutOfRange, utf16Offset, currentUTF16Offset)
	}

	// byteOffset now holds the byte position corresponding to the start of the
	// UTF-16 character at utf16Offset
	return byteOffset, nil
}

// byteOffsetToLSPPosition converts a 0-based byte offset to 0-based LSP line/char (UTF-16).
// (Moved from lsp_server.go, now in utils)
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

	// Now calculate the UTF-16 offset within the target line
	lineContentBytes := content[currentLineStartByteOffset:targetByteOffset]
	utf16CharOffset, convErr := bytesToUTF16Offset(lineContentBytes, logger)
	if convErr != nil {
		// Log the error but return the best guess (byte length as fallback)
		logger.Error("Error converting line bytes to UTF16 offset", "error", convErr, "line", currentLine)
		// Fallback: Use byte length as character count (less accurate for multi-byte)
		// This might be better than returning 0 or erroring out completely.
		utf16CharOffset = len(lineContentBytes)
	}

	return currentLine, uint32(utf16CharOffset), nil
}

// bytesToUTF16Offset calculates the number of UTF-16 code units for a byte slice.
// (Moved from lsp_server.go, now in utils)
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
			utf16Offset += 2 // Surrogate pair
		} else {
			utf16Offset += 1
		}
		byteOffset += size
	}
	return utf16Offset, nil
}

// ============================================================================
// Path Validation Helper (Defensive Programming)
// ============================================================================

// ValidateAndGetFilePath converts a file:// DocumentURI string to a clean, absolute local path.
// It returns an error if the URI scheme is not 'file' or parsing/cleaning fails.
// Uses slog for logging warnings.
// ** MODIFIED: Cycle 2 - Removed unused logger variable **
func ValidateAndGetFilePath(uri string) (string, error) {
	// Use default logger directly.
	// logger := slog.Default() // <-- This line should be gone or commented out

	if uri == "" {
		// Log here to ensure logger is used before potential early return
		slog.Warn("ValidateAndGetFilePath called with empty URI") // Use slog.Default() directly
		return "", errors.New("document URI cannot be empty")
	}
	// Log the entry point after the empty check
	slog.Debug("Validating URI", "uri", uri) // Use slog.Default() directly

	parsedURL, err := url.Parse(uri)
	if err != nil {
		slog.Warn("Failed to parse document URI", "uri", uri, "error", err) // Use slog.Default() directly
		return "", fmt.Errorf("%w: invalid document URI '%s': %w", ErrInvalidURI, uri, err)
	}

	// --- Security Check: Ensure scheme is 'file' ---
	if parsedURL.Scheme != "file" {
		slog.Warn("Received non-file document URI", "uri", uri, "scheme", parsedURL.Scheme) // Use slog.Default() directly
		return "", fmt.Errorf("%w: unsupported URI scheme: '%s' (only 'file://' is supported)", ErrInvalidURI, parsedURL.Scheme)
	}

	// Get path from URL (handles URL decoding like %20)
	filePath := parsedURL.Path

	// --- Platform-Specific Path Cleaning --- REMOVED Windows specific logic ---
	// On Unix-like systems (Linux, macOS), the path should generally be correct after url.Parse.

	// For local files, Host should be empty. If Host is present, it's likely not a local file path.
	if parsedURL.Host != "" {
		// Allow 'localhost' as a host for file URIs (some clients might send this)
		if strings.ToLower(parsedURL.Host) != "localhost" {
			slog.Warn("File URI includes unexpected host component", "uri", uri, "host", parsedURL.Host) // Use slog.Default() directly
			return "", fmt.Errorf("%w: file URI should not contain host component for local files (host: %s)", ErrInvalidURI, parsedURL.Host)
		}
	}

	// --- Security Check: Ensure absolute and clean path ---
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		slog.Warn("Failed to get absolute path", "uri_path", filePath, "error", err) // Use slog.Default() directly
		return "", fmt.Errorf("failed to resolve absolute path for '%s': %w", filePath, err)
	}

	slog.Debug("Validated and converted URI to path", "uri", uri, "path", absPath) // Use slog.Default() directly
	return absPath, nil
}

// PathToURI converts an absolute file path to a file:// URI.
// Added as inverse of ValidateAndGetFilePath.
func PathToURI(absPath string) (string, error) {
	if !filepath.IsAbs(absPath) {
		return "", fmt.Errorf("path must be absolute: %s", absPath)
	}
	// Ensure consistent forward slashes for URI
	absPath = filepath.ToSlash(absPath)
	// --- Platform-Specific Path Cleaning --- REMOVED Windows specific logic ---
	// On Unix-like systems, the path usually starts with '/' already.

	// Construct the URL
	uri := url.URL{
		Scheme: "file",
		Path:   absPath,
		// Host can be omitted or set to "localhost" for file URIs
		// Host: "",
	}
	return uri.String(), nil
}

// ============================================================================
// Cache Helper Functions (Updated with slog)
// ============================================================================

// calculateGoModHash calculates the SHA256 hash of the go.mod file.
func calculateGoModHash(dir string) string {
	logger := slog.Default().With("dir", dir) // Use default logger, assumes initialized by caller

	goModPath := filepath.Join(dir, "go.mod")
	f, err := os.Open(goModPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			logger.Debug("go.mod not found, using 'no-gomod' hash")
			return "no-gomod" // Consistent hash value when go.mod doesn't exist
		}
		logger.Warn("Error reading go.mod for hashing", "path", goModPath, "error", err)
		return "read-error" // Consistent hash value on read error
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		logger.Warn("Error hashing go.mod content", "path", goModPath, "error", err)
		return "hash-error" // Consistent hash value on hash error
	}
	return hex.EncodeToString(h.Sum(nil))
}

// calculateInputHashes calculates hashes for go.mod, go.sum, and Go files.
// Uses pkg.CompiledGoFiles if available, otherwise scans the directory.
// Uses slog for logging.
func calculateInputHashes(dir string, pkg *packages.Package) (map[string]string, error) {
	hashes := make(map[string]string)
	filesToHash := make(map[string]struct{})
	logger := slog.Default().With("dir", dir) // Use default logger

	// Always check for go.mod/go.sum relative to the directory.
	for _, fname := range []string{"go.mod", "go.sum"} {
		fpath := filepath.Join(dir, fname)
		// Check if file exists before trying to hash
		if _, err := os.Stat(fpath); err == nil {
			// Get absolute path for consistency if needed, though relative might be okay here
			if absPath, absErr := filepath.Abs(fpath); absErr == nil {
				filesToHash[absPath] = struct{}{}
			} else {
				logger.Warn("Could not get absolute path for hashing", "file", fpath, "error", absErr)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			logger.Warn("Failed to stat file for hashing", "file", fpath, "error", err)
		}
	}

	// Use compiled files from package info if available and seems valid.
	filesFromPkg := false
	if pkg != nil && len(pkg.CompiledGoFiles) > 0 {
		// Check if CompiledGoFiles actually contains paths before trying to use the first one
		if len(pkg.CompiledGoFiles[0]) > 0 {
			firstFileAbs, _ := filepath.Abs(pkg.CompiledGoFiles[0])
			dirAbs, _ := filepath.Abs(dir)
			// Ensure first file is absolute and within the target directory
			if filepath.IsAbs(firstFileAbs) && strings.HasPrefix(firstFileAbs, dirAbs) {
				filesFromPkg = true
				logger.Debug("Hashing based on CompiledGoFiles", "count", len(pkg.CompiledGoFiles), "package", pkg.ID)
				for _, fpath := range pkg.CompiledGoFiles {
					// Ensure path is absolute before adding
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

	// Fallback: Scan directory for .go files if package info unavailable/empty/invalid.
	if !filesFromPkg {
		logger.Debug("Calculating input hashes by scanning directory (pkg info unavailable or invalid)")
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

	// Calculate hash for each unique absolute file path.
	for absPath := range filesToHash {
		relPath, err := filepath.Rel(dir, absPath)
		if err != nil {
			logger.Warn("Could not get relative path for hashing, using base name", "absPath", absPath, "error", err)
			relPath = filepath.Base(absPath)
		}
		relPath = filepath.ToSlash(relPath) // Ensure forward slashes in keys.

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
		return "", err // Propagate error (e.g., os.ErrNotExist)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err // Propagate hashing error
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// compareFileHashes compares current and cached file hashes, logging differences.
// Uses slog for logging.
func compareFileHashes(current, cached map[string]string) bool {
	logger := slog.Default() // Use default logger

	if len(current) != len(cached) {
		logger.Debug("Cache invalid: File count mismatch", "current_count", len(current), "cached_count", len(cached))
		// Log specific differences only if debug is enabled
		if logger.Enabled(context.Background(), slog.LevelDebug) {
			for relPath := range current {
				if _, ok := cached[relPath]; !ok {
					logger.Debug("File exists now but was not in cache", "file", relPath)
				}
			}
			for relPath := range cached {
				if _, ok := current[relPath]; !ok {
					logger.Debug("File was cached but does not exist now", "file", relPath)
				}
			}
		}
		return false
	}
	for relPath, currentHash := range current {
		cachedHash, ok := cached[relPath]
		if !ok {
			// This case should be caught by the length check above, but added for safety
			logger.Debug("Cache invalid: File missing in cache despite matching counts", "file", relPath)
			return false
		}
		if currentHash != cachedHash {
			logger.Debug("Cache invalid: Hash mismatch", "file", relPath)
			return false
		}
	}
	return true // Counts match and all hashes match.
}

// deleteCacheEntryByKey removes an entry directly using the key.
// Uses slog for logging.
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
			// Bucket not existing isn't an error for deletion, just means key isn't there
			return nil
		}
		// Check if key exists before attempting delete (optional, Delete is idempotent)
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
// Retry Helper (Updated with slog)
// ============================================================================

// retry executes an operation function with backoff and retry logic.
// Uses slog for logging. Requires logger to be passed.
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration, logger *slog.Logger) error {
	var lastErr error
	if logger == nil {
		logger = slog.Default() // Use default if nil
	}

	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		attemptLogger := logger.With("attempt", i+1, "max_attempts", maxRetries)
		select {
		case <-ctx.Done():
			attemptLogger.Warn("Context cancelled before attempt", "error", ctx.Err())
			return ctx.Err() // Return context error immediately
		default:
		}

		lastErr = operation()
		if lastErr == nil {
			return nil // Success
		}

		// --- Error Classification ---
		// Don't retry context errors
		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			attemptLogger.Warn("Attempt failed due to context error. Not retrying.", "error", lastErr)
			return lastErr
		}

		// Check for specific retryable Ollama errors
		var ollamaErr *OllamaError
		isRetryableOllama := errors.As(lastErr, &ollamaErr) &&
			(ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests || ollamaErr.Status == http.StatusInternalServerError)

		// Check for general network/connection errors that might be temporary
		isRetryableNetwork := errors.Is(lastErr, ErrOllamaUnavailable) || errors.Is(lastErr, ErrStreamProcessing)
		// Add more specific network error checks if needed (e.g., net.OpError, syscall errors)

		isRetryable := isRetryableOllama || isRetryableNetwork

		if !isRetryable {
			attemptLogger.Warn("Attempt failed with non-retryable error.", "error", lastErr)
			return lastErr // Return the original non-retryable error
		}

		// If it's the last attempt, break the loop and return the last error
		if i == maxRetries-1 {
			break
		}

		waitDuration := currentDelay
		attemptLogger.Warn("Attempt failed with retryable error. Retrying...", "error", lastErr, "delay", waitDuration)

		// Wait for the delay, respecting context cancellation
		select {
		case <-ctx.Done():
			attemptLogger.Warn("Context cancelled during retry wait", "error", ctx.Err())
			return ctx.Err() // Return context error immediately
		case <-time.After(waitDuration):
			// Optionally increase delay for next attempt (exponential backoff)
			// currentDelay *= 2
			// if currentDelay > maxDelay { currentDelay = maxDelay } // Cap delay
		}
	}
	// If loop finished, it means all retries failed
	logger.Error("Operation failed after all retries.", "retries", maxRetries, "final_error", lastErr)
	// Return the last error encountered, wrapped for context
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr)
}

// ============================================================================
// Spinner (Remains unchanged, uses direct fmt printing)
// ============================================================================

// Spinner provides simple terminal spinner feedback.
type Spinner struct {
	chars    []string
	message  string
	index    int
	mu       sync.Mutex
	stopChan chan struct{}
	doneChan chan struct{}
	running  bool
}

func NewSpinner() *Spinner {
	return &Spinner{chars: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}, index: 0}
}

// Start begins the spinner animation in a separate goroutine.
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
			// Clear the spinner line only if it was actually running
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
				// Check running flag again inside loop
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

// UpdateMessage changes the text displayed next to the spinner.
func (s *Spinner) UpdateMessage(newMessage string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		s.message = newMessage
	}
}

// Stop halts the spinner animation and cleans up.
func (s *Spinner) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	// Close stopChan only if it hasn't been closed already
	select {
	case <-s.stopChan:
		// Already closed
	default:
		close(s.stopChan)
	}
	doneChan := s.doneChan // Read doneChan while holding lock
	s.mu.Unlock()

	// Wait for the goroutine to finish, with a timeout
	if doneChan != nil {
		select {
		case <-doneChan:
			// Goroutine finished cleanly
		case <-time.After(500 * time.Millisecond):
			slog.Warn("Timeout waiting for spinner goroutine cleanup")
		}
	}
	// Ensure the line is cleared after stopping
	fmt.Fprintf(os.Stderr, "\r\033[K")
}

// ============================================================================
// Snippet Extraction Helper (Updated with slog)
// ============================================================================

// SnippetContext holds the code prefix and suffix relative to the cursor.
type SnippetContext struct {
	Prefix   string // Code before cursor.
	Suffix   string // Code after cursor.
	FullLine string // Full line where cursor is located.
}

// extractSnippetContext extracts code prefix, suffix, and full line around the cursor.
// Expects filename to be an absolute, validated path. Uses slog.
func extractSnippetContext(filename string, row, col int) (SnippetContext, error) {
	logger := slog.Default().With("file", filename, "line", row, "col", col) // Use default logger
	var ctx SnippetContext

	contentBytes, err := os.ReadFile(filename) // Read validated path
	if err != nil {
		return ctx, fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	content := string(contentBytes)

	fset := token.NewFileSet()
	// AddFile base is 1 to ensure all positions are > 0
	file := fset.AddFile(filename, 1, len(contentBytes))
	if file == nil {
		// This should be unlikely if filename is valid
		return ctx, fmt.Errorf("failed to add file '%s' to fileset", filename)
	}

	cursorPos, posErr := calculateCursorPos(file, row, col) // Uses slog internally
	if posErr != nil {
		// calculateCursorPos already logged details if needed
		return ctx, fmt.Errorf("cannot determine valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		// Should be caught by posErr check, but added for safety
		return ctx, fmt.Errorf("%w: invalid cursor position calculated (Pos: %d)", ErrPositionConversion, cursorPos)
	}

	offset := file.Offset(cursorPos)
	// Clamp offset to valid range [0, len(content)]
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
		// Return prefix/suffix even if full line fails
		return ctx, nil
	}
	startOffset := file.Offset(lineStartPos)

	// Find end of line (start of next line, or EOF)
	lineEndOffset := file.Size() // Default to EOF
	fileLineCount := file.LineCount()
	if row < fileLineCount {
		nextLineStartPos := file.LineStart(row + 1)
		if nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		} else {
			// This case might occur if the file structure is unusual
			logger.Warn("Could not get start position for next line", "line", row+1)
		}
	}

	// Ensure offsets are valid before slicing
	if startOffset >= 0 && lineEndOffset >= startOffset && lineEndOffset <= len(content) {
		lineContent := content[startOffset:lineEndOffset]
		// Trim trailing newline characters (\n or \r\n)
		lineContent = strings.TrimRight(lineContent, "\n")
		lineContent = strings.TrimRight(lineContent, "\r")
		ctx.FullLine = lineContent
	} else {
		logger.Warn("Could not extract full line content due to invalid offsets",
			"startOffset", startOffset,
			"lineEndOffset", lineEndOffset,
			"contentLen", len(content))
	}
	return ctx, nil
}

// calculateCursorPos converts 1-based line/col to 0-based token.Pos offset.
// This function is now defined here in utils, accessible within the package.
// Uses slog for logging warnings about clamping.
func calculateCursorPos(file *token.File, line, col int) (token.Pos, error) {
	logger := slog.Default() // Use default logger
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
	// Allow cursor to be at the start of the line immediately after the last line
	if line > fileLineCount+1 || (line == fileLineCount+1 && col > 1) {
		return token.NoPos, fmt.Errorf("%w: line number %d / column %d exceeds file bounds (%d lines)", ErrPositionOutOfRange, line, col, fileLineCount)
	}

	// Handle case where cursor is on the virtual line after the last line
	if line == fileLineCount+1 {
		if col == 1 {
			return file.Pos(file.Size()), nil // Position at EOF
		}
		// Should have been caught by the check above, but added for safety
		return token.NoPos, fmt.Errorf("%w: column %d invalid on virtual line %d (only col 1 allowed)", ErrPositionOutOfRange, col, line)
	}

	// Cursor is within the actual lines of the file
	lineStartPos := file.LineStart(line)
	if !lineStartPos.IsValid() {
		// This might happen if the file is empty or line number is invalid (though checked above)
		return token.NoPos, fmt.Errorf("%w: cannot get start offset for line %d in file '%s'", ErrPositionConversion, line, file.Name())
	}

	// Calculate the target offset based on 1-based column
	lineStartOffset := file.Offset(lineStartPos)
	targetOffset := lineStartOffset + col - 1

	// Find the end offset of the current line (start of next line or EOF)
	lineEndOffset := file.Size()
	if line < fileLineCount {
		nextLineStartPos := file.LineStart(line + 1)
		if nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
		// If next line start is invalid, lineEndOffset remains file.Size()
	}

	// Clamp the target offset to the valid range [lineStartOffset, lineEndOffset]
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

	// Convert final (potentially clamped) offset to token.Pos
	pos := file.Pos(finalOffset)
	if !pos.IsValid() {
		// If the clamped offset still results in an invalid Pos, something is wrong.
		// Fallback to line start position.
		logger.Error("Clamped offset resulted in invalid token.Pos. Falling back to line start.", "offset", finalOffset, "line_start_pos", lineStartPos)
		return lineStartPos, fmt.Errorf("%w: failed to calculate valid token.Pos for offset %d (clamped: %v)", ErrPositionConversion, finalOffset, clamped)
	}
	return pos, nil
}

// ============================================================================
// Stream Processing Helpers (Updated with slog)
// ============================================================================

// streamCompletion reads the Ollama stream response and writes it to w.
// Uses slog for logging.
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer) error {
	defer r.Close()
	reader := bufio.NewReader(r)
	lineCount := 0
	logger := slog.Default() // Use default logger

	for {
		select {
		case <-ctx.Done():
			logger.Warn("Context cancelled during streaming", "error", ctx.Err())
			return ctx.Err() // Return context error immediately
		default:
		}

		line, err := reader.ReadBytes('\n')

		// Process the line content *before* checking EOF, as EOF might occur
		// after reading the last chunk without a trailing newline.
		if len(line) > 0 {
			if procErr := processLine(line, w, logger); procErr != nil {
				return procErr // Return processing error immediately
			}
			lineCount++
		}

		// Check for errors *after* processing potential data
		if err != nil {
			if err == io.EOF {
				logger.Debug("Stream processing finished (EOF)", "lines_processed", lineCount)
				return nil // Clean EOF
			}
			// Check context cancellation again after read error
			select {
			case <-ctx.Done():
				return ctx.Err() // Prioritize context error
			default:
				// Return the original read error
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			}
		}
	}
}

// processLine decodes a single line from the Ollama stream and writes the content.
// Uses slog for logging.
func processLine(line []byte, w io.Writer, logger *slog.Logger) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil // Ignore empty lines
	}

	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		// Log unexpected non-JSON lines if debugging
		logger.Debug("Ignoring non-JSON line from Ollama stream", "line", string(line))
		return nil // Don't treat as fatal, just ignore
	}

	// Check for errors reported within the JSON payload
	if resp.Error != "" {
		logger.Error("Ollama stream reported an error", "error", resp.Error)
		// Wrap the Ollama error message for better context
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	}

	// Write the actual response chunk to the output writer
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		// Log and return error if writing fails
		logger.Error("Error writing stream chunk to output", "error", err)
		return fmt.Errorf("error writing to output: %w", err)
	}
	return nil // Success
}

// Helper function to truncate strings for logging (moved from lsp_server.go)
func firstN(s string, n int) string {
	if n < 0 {
		n = 0
	}
	count := 0
	for i := range s {
		if count == n {
			// Add ellipsis if truncated
			return s[:i] + "..."
		}
		count++
	}
	// Return original string if length <= n
	return s
}
