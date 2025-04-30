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
	"log/slog" // Cycle 3: Use slog
	"net/http" // Needed for OllamaError status codes
	"net/url"  // Added for URI parsing
	"os"
	"path/filepath"
	"runtime" // Added for OS-specific path handling
	"strings"
	"sync"
	"time"
	"unicode/utf8" // For UTF-16 conversion

	"go.etcd.io/bbolt" // Needed for cache helpers
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

// NOTE: LSPPosition struct definition removed from here.
// The definition in lsp_protocol.go is now used package-wide.

// LspPositionToBytePosition converts 0-based LSP line/character (UTF-16) to
// 1-based Go line/column (bytes) and 0-based byte offset.
// Uses slog for logging warnings.
// Uses the LSPPosition type defined in lsp_protocol.go (implicitly within the package).
func LspPositionToBytePosition(content []byte, lspPos LSPPosition) (line, col, byteOffset int, err error) {
	// Use default logger, assumes initialized by caller (main/test)
	logger := slog.Default()

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
				if errors.Is(convErr, ErrPositionOutOfRange) { // Clamp to line end on out-of-range error.
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
	// Use default logger, assumes initialized by caller (main/test)
	logger := slog.Default()

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
		logger.Warn("UTF16 offset clamping required",
			"requested_utf16", utf16Offset,
			"actual_utf16_len", currentUTF16Offset,
			"byte_len", lineLenBytes)
		// Return the byte offset corresponding to the end of the line
		return lineLenBytes, fmt.Errorf("%w: utf16Offset %d is beyond the line length in UTF-16 units (%d)", ErrPositionOutOfRange, utf16Offset, currentUTF16Offset)
	}

	// byteOffset now holds the byte position corresponding to the start of the
	// UTF-16 character at utf16Offset
	return byteOffset, nil
}

// ============================================================================
// Path Validation Helper (Defensive Programming)
// ============================================================================

// ValidateAndGetFilePath converts a file:// DocumentURI string to a clean, absolute local path.
// It returns an error if the URI scheme is not 'file' or parsing/cleaning fails.
// Uses slog for logging warnings.
func ValidateAndGetFilePath(uri string, logger *slog.Logger) (string, error) {
	// Use provided logger or default if nil
	if logger == nil {
		logger = slog.Default()
	}

	if uri == "" {
		return "", errors.New("document URI cannot be empty")
	}
	parsedURL, err := url.Parse(uri) // Use standard string directly
	if err != nil {
		logger.Warn("Failed to parse document URI", "uri", uri, "error", err)
		return "", fmt.Errorf("%w: invalid document URI '%s': %w", ErrInvalidURI, uri, err)
	}

	// --- Security Check: Ensure scheme is 'file' ---
	if parsedURL.Scheme != "file" {
		logger.Warn("Received non-file document URI", "uri", uri, "scheme", parsedURL.Scheme)
		return "", fmt.Errorf("%w: unsupported URI scheme: '%s' (only 'file://' is supported)", ErrInvalidURI, parsedURL.Scheme)
	}

	// Get path from URL (handles URL decoding like %20)
	filePath := parsedURL.Path

	// --- Platform-Specific Path Cleaning ---
	// On Windows, paths might start with "/C:/..." after parsing. Remove the leading slash.
	if runtime.GOOS == "windows" {
		if len(filePath) > 2 && filePath[0] == '/' && filePath[2] == ':' { // Checks for /X:/ pattern
			filePath = filePath[1:]
		}
		// Windows paths might also be represented as just "C:/..." in Path
		// filepath.Abs should handle this correctly.
	}
	// For other OS, url.Parse might produce "//hostname/path". We only want the path.
	// However, for local files, Host should be empty. If Host is present, it's likely not a local file path.
	if parsedURL.Host != "" {
		logger.Warn("File URI includes unexpected host component", "uri", uri, "host", parsedURL.Host)
		return "", fmt.Errorf("%w: file URI should not contain host component for local files", ErrInvalidURI)
	}

	// --- Security Check: Ensure absolute and clean path ---
	// filepath.Abs also cleans the path (e.g., removes ../, resolves symlinks conceptually)
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		logger.Warn("Failed to get absolute path", "uri_path", filePath, "error", err)
		return "", fmt.Errorf("failed to resolve absolute path for '%s': %w", filePath, err)
	}

	logger.Debug("Validated and converted URI to path", "uri", uri, "path", absPath)
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
	// On Windows, add a leading slash if it's missing (e.g., C:/ -> /C:/)
	if runtime.GOOS == "windows" && len(absPath) > 1 && absPath[1] == ':' {
		absPath = "/" + absPath
	}
	// Construct the URL
	uri := url.URL{
		Scheme: "file",
		Path:   absPath,
	}
	return uri.String(), nil
}

// ============================================================================
// Cache Helper Functions (Updated with slog)
// ============================================================================

// calculateGoModHash calculates the SHA256 hash of the go.mod file.
func calculateGoModHash(dir string) string {
	// Use default logger, assumes initialized by caller (main/test)
	logger := slog.Default().With("dir", dir)

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
		firstFileAbs, _ := filepath.Abs(pkg.CompiledGoFiles[0])
		dirAbs, _ := filepath.Abs(dir)
		if len(pkg.CompiledGoFiles[0]) > 0 && filepath.IsAbs(firstFileAbs) && strings.HasPrefix(firstFileAbs, dirAbs) {
			filesFromPkg = true
			logger.Debug("Hashing based on CompiledGoFiles", "count", len(pkg.CompiledGoFiles), "package", pkg.ID)
			for _, fpath := range pkg.CompiledGoFiles {
				filesToHash[fpath] = struct{}{}
			}
		} else {
			logger.Warn("CompiledGoFiles paths seem invalid or not absolute/relative to dir, falling back to directory scan.",
				"first_file", pkg.CompiledGoFiles[0], "dir", dir)
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
// Uses slog for logging.
func compareFileHashes(current, cached map[string]string) bool {
	logger := slog.Default() // Use default logger

	if len(current) != len(cached) {
		logger.Debug("Cache invalid: File count mismatch", "current_count", len(current), "cached_count", len(cached))
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
// Retry Helper (Updated with slog)
// ============================================================================

// retry executes an operation function with backoff and retry logic.
// Uses slog for logging. Requires logger to be passed.
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration, logger *slog.Logger) error {
	var lastErr error
	if logger == nil {
		logger = slog.Default()
	}

	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		attemptLogger := logger.With("attempt", i+1, "max_attempts", maxRetries)
		select {
		case <-ctx.Done():
			attemptLogger.Warn("Context cancelled before attempt", "error", ctx.Err())
			return ctx.Err()
		default:
		}

		lastErr = operation()
		if lastErr == nil {
			return nil
		}

		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			attemptLogger.Warn("Attempt failed due to context error. Not retrying.", "error", lastErr)
			return lastErr
		}

		var ollamaErr *OllamaError
		isRetryable := errors.As(lastErr, &ollamaErr) && (ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		isRetryable = isRetryable || errors.Is(lastErr, ErrOllamaUnavailable)

		if !isRetryable {
			attemptLogger.Warn("Attempt failed with non-retryable error.", "error", lastErr)
			return lastErr
		}

		if i == maxRetries-1 {
			break
		}

		waitDuration := currentDelay
		attemptLogger.Warn("Attempt failed with retryable error. Retrying...", "error", lastErr, "delay", waitDuration)

		select {
		case <-ctx.Done():
			attemptLogger.Warn("Context cancelled during retry wait", "error", ctx.Err())
			return ctx.Err()
		case <-time.After(waitDuration):
			// Optionally increase delay: currentDelay *= 2
		}
	}
	logger.Error("Operation failed after all retries.", "retries", maxRetries, "final_error", lastErr)
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
			slog.Warn("Timeout waiting for spinner goroutine cleanup")
		}
	}
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
	file := fset.AddFile(filename, 1, len(contentBytes))
	if file == nil {
		return ctx, fmt.Errorf("failed to add file '%s' to fileset", filename)
	}

	cursorPos, posErr := calculateCursorPos(file, row, col) // Uses slog internally
	if posErr != nil {
		return ctx, fmt.Errorf("cannot determine valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		return ctx, fmt.Errorf("%w: invalid cursor position calculated (Pos: %d)", ErrPositionConversion, cursorPos)
	}

	offset := file.Offset(cursorPos)
	if offset < 0 {
		offset = 0
	}
	if offset > len(content) {
		offset = len(content)
	}

	ctx.Prefix = content[:offset]
	ctx.Suffix = content[offset:]

	lineStartPos := file.LineStart(row)
	if !lineStartPos.IsValid() {
		logger.Warn("Could not get start position for line")
		return ctx, nil
	}
	startOffset := file.Offset(lineStartPos)

	lineEndOffset := file.Size()
	fileLineCount := file.LineCount()
	if row < fileLineCount {
		if nextLineStartPos := file.LineStart(row + 1); nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	if startOffset >= 0 && lineEndOffset >= startOffset && lineEndOffset <= len(content) {
		lineContent := content[startOffset:lineEndOffset]
		lineContent = strings.TrimRight(lineContent, "\n")
		lineContent = strings.TrimRight(lineContent, "\r")
		ctx.FullLine = lineContent
	} else {
		logger.Warn("Could not extract full line content based on offsets",
			"startOffset", startOffset,
			"lineEndOffset", lineEndOffset,
			"contentLen", len(content))
	}
	return ctx, nil
}

// calculateCursorPos converts 1-based line/col to 0-based token.Pos offset.
// This function is now defined here in utils, accessible within the package.
func calculateCursorPos(file *token.File, line, col int) (token.Pos, error) {
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
	if line > fileLineCount {
		if line == fileLineCount+1 && col == 1 {
			return file.Pos(file.Size()), nil
		}
		return token.NoPos, fmt.Errorf("%w: line number %d exceeds file line count %d", ErrPositionOutOfRange, line, fileLineCount)
	}

	lineStartPos := file.LineStart(line)
	if !lineStartPos.IsValid() {
		return token.NoPos, fmt.Errorf("%w: cannot get start offset for line %d in file '%s'", ErrPositionConversion, line, file.Name())
	}

	lineStartOffset := file.Offset(lineStartPos)
	cursorOffset := lineStartOffset + col - 1

	lineEndOffset := file.Size()
	if line < fileLineCount {
		if nextLineStartPos := file.LineStart(line + 1); nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	finalOffset := cursorOffset
	if cursorOffset < lineStartOffset {
		finalOffset = lineStartOffset
		slog.Warn("Column resulted in offset before line start. Clamping.", "col", col, "offset", cursorOffset, "line_start", lineStartOffset)
	}
	if cursorOffset > lineEndOffset {
		finalOffset = lineEndOffset
		slog.Warn("Column resulted in offset beyond line end. Clamping.", "col", col, "offset", cursorOffset, "line_end", lineEndOffset)
	}

	pos := file.Pos(finalOffset)
	if !pos.IsValid() {
		slog.Error("Clamped offset resulted in invalid token.Pos. Using line start.", "offset", finalOffset, "line_start_pos", lineStartPos)
		return lineStartPos, fmt.Errorf("%w: failed to calculate valid token.Pos for offset %d", ErrPositionConversion, finalOffset)
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
			return ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')

		if err != nil {
			if err == io.EOF {
				if len(line) > 0 {
					if procErr := processLine(line, w, logger); procErr != nil {
						return procErr
					}
				}
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

		lineCount++
		if procErr := processLine(line, w, logger); procErr != nil {
			return procErr
		}
	}
}

// processLine decodes a single line from the Ollama stream and writes the content.
// Uses slog for logging.
func processLine(line []byte, w io.Writer, logger *slog.Logger) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil
	}

	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		logger.Debug("Ignoring non-JSON line from Ollama stream", "line", string(line))
		return nil
	}

	if resp.Error != "" {
		logger.Error("Ollama stream reported an error", "error", resp.Error)
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	}

	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		logger.Error("Error writing stream chunk to output", "error", err)
		return fmt.Errorf("error writing to output: %w", err)
	}
	return nil
}

// Helper function to truncate strings for logging (moved from lsp_server.go)
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
