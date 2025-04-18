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
	"log/slog" // Cycle 3: Added slog
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"go.etcd.io/bbolt" // Needed for cache helpers
	"golang.org/x/tools/go/packages"
)

// ============================================================================
// Terminal Colors
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
// Exported Helper Functions
// ============================================================================

// PrettyPrint prints colored text to stderr.
func PrettyPrint(color, text string) {
	// Consider using slog for this if structure is desired,
	// but direct stderr printing is fine for simple colored output.
	fmt.Fprint(os.Stderr, color, text, ColorReset)
}

// ============================================================================
// LSP Position Conversion Helpers
// ============================================================================

// LSPPosition represents a 0-based line/character offset (UTF-16).
// Duplicated from lsp main for use in core library without circular dependency
type LSPPosition struct {
	Line      uint32 `json:"line"`      // 0-based
	Character uint32 `json:"character"` // 0-based, UTF-16 offset
}

// LspPositionToBytePosition converts 0-based LSP line/character (UTF-16) to
// 1-based Go line/column (bytes) and 0-based byte offset.
func LspPositionToBytePosition(content []byte, lspPos LSPPosition) (line, col, byteOffset int, err error) {
	if content == nil {
		return 0, 0, -1, fmt.Errorf("%w: file content is nil", ErrPositionConversion)
	}
	targetLine := int(lspPos.Line)
	targetUTF16Char := int(lspPos.Character)
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
		newlineLengthBytes := 1 // Assume \n for simplicity
		if currentLine == targetLine {
			byteOffsetInLine, convErr := Utf16OffsetToBytes(lineTextBytes, targetUTF16Char)
			if convErr != nil {
				if errors.Is(convErr, ErrPositionOutOfRange) { // Clamp to line end on out-of-range error.
					// Cycle 3: Use slog
					slog.Warn("UTF16 offset out of range, clamping to line end",
						"line", targetLine,
						"char", targetUTF16Char,
						"error", convErr)
					byteOffsetInLine = lineLengthBytes
				} else {
					return 0, 0, -1, fmt.Errorf("failed converting UTF16 to byte offset on line %d: %w", currentLine, convErr)
				}
			}
			line = currentLine + 1
			col = byteOffsetInLine + 1
			byteOffset = currentByteOffset + byteOffsetInLine
			return line, col, byteOffset, nil // Success.
		}
		currentByteOffset += lineLengthBytes + newlineLengthBytes
		currentLine++
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, -1, fmt.Errorf("%w: error scanning file content: %w", ErrPositionConversion, err)
	}

	// Handle cursor on the line after the last line of content.
	if currentLine == targetLine {
		if targetUTF16Char == 0 {
			line = currentLine + 1
			col = 1
			byteOffset = currentByteOffset
			return line, col, byteOffset, nil
		}
		return 0, 0, -1, fmt.Errorf("%w: invalid character offset %d on line %d (after last line with content)", ErrPositionOutOfRange, targetUTF16Char, targetLine)
	}
	// Target line not found.
	return 0, 0, -1, fmt.Errorf("%w: LSP line %d not found in file (total lines scanned %d)", ErrPositionOutOfRange, targetLine, currentLine)
}

// Utf16OffsetToBytes converts a 0-based UTF-16 offset within a line to a 0-based byte offset.
func Utf16OffsetToBytes(line []byte, utf16Offset int) (int, error) {
	if utf16Offset < 0 {
		return 0, fmt.Errorf("%w: invalid utf16Offset: %d (must be >= 0)", ErrInvalidPositionInput, utf16Offset)
	}
	if utf16Offset == 0 {
		return 0, nil
	}

	byteOffset := 0
	currentUTF16Offset := 0
	for byteOffset < len(line) {
		if currentUTF16Offset >= utf16Offset {
			break
		} // Reached target.
		r, size := utf8.DecodeRune(line[byteOffset:])
		if r == utf8.RuneError && size <= 1 {
			return byteOffset, fmt.Errorf("%w at byte offset %d", ErrInvalidUTF8, byteOffset)
		}
		utf16Units := 1
		if r > 0xFFFF {
			utf16Units = 2
		} // Surrogate pairs require 2 units.
		// If adding this rune exceeds target, current byteOffset is the answer.
		if currentUTF16Offset+utf16Units > utf16Offset {
			break
		}
		currentUTF16Offset += utf16Units
		byteOffset += size
		if currentUTF16Offset == utf16Offset {
			break
		} // Exact match.
	}
	// Check if target offset was beyond the actual line length in UTF-16.
	if currentUTF16Offset < utf16Offset {
		return len(line), fmt.Errorf("%w: utf16Offset %d is beyond the line length in UTF-16 units (%d)", ErrPositionOutOfRange, utf16Offset, currentUTF16Offset)
	}
	return byteOffset, nil
}

// ============================================================================
// Cache Helper Functions
// ============================================================================

// calculateGoModHash calculates the SHA256 hash of the go.mod file.
func calculateGoModHash(dir string) string {
	goModPath := filepath.Join(dir, "go.mod")
	f, err := os.Open(goModPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "no-gomod"
		}
		// Cycle 3: Use slog
		slog.Warn("Error reading go.mod for hashing", "path", goModPath, "error", err)
		return "read-error"
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		// Cycle 3: Use slog
		slog.Warn("Error hashing go.mod content", "path", goModPath, "error", err)
		return "hash-error"
	}
	return hex.EncodeToString(h.Sum(nil))
}

// calculateInputHashes calculates hashes for go.mod, go.sum, and Go files.
// Uses pkg.CompiledGoFiles if available, otherwise scans the directory (non-recursive).
func calculateInputHashes(dir string, pkg *packages.Package) (map[string]string, error) {
	hashes := make(map[string]string)
	filesToHash := make(map[string]struct{})
	// Cycle 3: Create contextual logger
	logger := slog.Default().With("dir", dir)

	// Always check for go.mod/go.sum.
	for _, fname := range []string{"go.mod", "go.sum"} {
		fpath := filepath.Join(dir, fname)
		if _, err := os.Stat(fpath); err == nil {
			if absPath, absErr := filepath.Abs(fpath); absErr == nil {
				filesToHash[absPath] = struct{}{}
			} else {
				logger.Warn("Could not get absolute path for hashing", "file", fpath, "error", absErr)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("failed to stat %s: %w", fpath, err)
		}
	}

	// Phase 2, Step 4: Use compiled files from package info if available.
	filesFromPkg := false
	if pkg != nil && len(pkg.CompiledGoFiles) > 0 {
		filesFromPkg = true
		logger.Debug("Hashing based on CompiledGoFiles", "count", len(pkg.CompiledGoFiles), "package", pkg.PkgPath)
		for _, fpath := range pkg.CompiledGoFiles {
			// CompiledGoFiles should already be absolute paths.
			filesToHash[fpath] = struct{}{}
		}
	}

	// Fallback: Scan directory for .go files if package info unavailable/empty.
	if !filesFromPkg {
		logger.Debug("Calculating input hashes by scanning directory (pkg info unavailable or empty)")
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, fmt.Errorf("failed to scan directory %s for hashing: %w", dir, err)
		}
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") {
				absPath := filepath.Join(dir, entry.Name())
				filesToHash[absPath] = struct{}{}
			}
		}
	}

	// Calculate hash for each unique file path.
	for absPath := range filesToHash {
		relPath, err := filepath.Rel(dir, absPath)
		if err != nil {
			logger.Warn("Could not get relative path for hashing, using base name", "absPath", absPath, "error", err)
			relPath = filepath.Base(absPath)
		}
		relPath = filepath.ToSlash(relPath) // Use forward slashes for consistent keys.

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
func compareFileHashes(current, cached map[string]string) bool {
	if len(current) != len(cached) {
		// Cycle 3: Use slog
		slog.Debug("Cache invalid: File count mismatch", "current_count", len(current), "cached_count", len(cached))
		// Log specific differences at Debug level
		if slog.Default().Enabled(context.Background(), slog.LevelDebug) {
			for relPath := range current {
				if _, ok := cached[relPath]; !ok {
					slog.Debug("File exists now but was not in cache", "file", relPath)
				}
			}
			for relPath := range cached {
				if _, ok := current[relPath]; !ok {
					slog.Debug("File was cached but does not exist now", "file", relPath)
				}
			}
		}
		return false
	}
	for relPath, currentHash := range current {
		cachedHash, ok := cached[relPath]
		if !ok {
			// Cycle 3: Use slog
			slog.Debug("Cache invalid: File missing in cache despite matching counts", "file", relPath)
			return false
		} // Should not happen if counts match.
		if currentHash != cachedHash {
			// Cycle 3: Use slog
			slog.Debug("Cache invalid: Hash mismatch", "file", relPath)
			return false
		}
	}
	return true // Counts match and all hashes match.
}

// deleteCacheEntryByKey removes an entry directly using the key.
// Cycle 3: Added logger argument
func deleteCacheEntryByKey(db *bbolt.DB, cacheKey []byte, logger *slog.Logger) error {
	if db == nil {
		return errors.New("cannot delete cache entry: db is nil")
	}
	// Use provided logger or default if nil
	if logger == nil {
		logger = slog.Default()
	}
	logger = logger.With("cache_key", string(cacheKey)) // Add context

	err := db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(cacheBucketName)
		if b == nil {
			logger.Warn("Cache bucket not found during delete attempt.")
			return nil // Bucket gone? Nothing to do.
		}
		if b.Get(cacheKey) == nil {
			logger.Debug("Cache key not found during delete attempt.")
			return nil // Key doesn't exist.
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
// Cycle 3: Added logger argument
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration, logger *slog.Logger) error {
	var lastErr error
	// Use provided logger or default if nil
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
		} // Check context.

		lastErr = operation()
		if lastErr == nil {
			return nil
		} // Success.

		// Don't retry context errors.
		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			attemptLogger.Warn("Attempt failed due to context error. Not retrying.", "error", lastErr)
			return lastErr
		}

		// Check for other retryable errors (e.g., specific HTTP statuses from OllamaError)
		var ollamaErr *OllamaError
		isRetryable := errors.As(lastErr, &ollamaErr) && (ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		isRetryable = isRetryable || errors.Is(lastErr, ErrOllamaUnavailable) // Include general unavailability

		if !isRetryable {
			attemptLogger.Warn("Attempt failed with non-retryable error.", "error", lastErr)
			return lastErr
		}

		// If it's the last attempt, don't wait, just return the error.
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
		} // Wait or cancel.
	}
	logger.Error("Operation failed after all retries.", "retries", maxRetries, "final_error", lastErr)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr) // Return last error after retries exhausted.
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
			select {
			case s.doneChan <- struct{}{}:
			default:
			}
			close(s.doneChan)
		}() // Cleanup.
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
		} // Animate.
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
			// Cycle 3: Use slog
			slog.Warn("Timeout waiting for spinner goroutine cleanup")
		}
	}
	fmt.Fprintf(os.Stderr, "\r\033[K") // Wait & final clear.
}

// ============================================================================
// Snippet Extraction Helper
// ============================================================================

// SnippetContext holds the code prefix and suffix relative to the cursor.
type SnippetContext struct {
	Prefix   string // Code before cursor.
	Suffix   string // Code after cursor.
	FullLine string // Full line where cursor is located.
}

// extractSnippetContext extracts code prefix, suffix, and full line around the cursor.
func extractSnippetContext(filename string, row, col int) (SnippetContext, error) {
	var ctx SnippetContext
	contentBytes, err := os.ReadFile(filename)
	if err != nil {
		return ctx, fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	content := string(contentBytes)

	fset := token.NewFileSet()
	// Base must be 1 for LineStart to work correctly with 1-based line numbers.
	file := fset.AddFile(filename, 1, len(contentBytes))
	if file == nil {
		return ctx, fmt.Errorf("failed to add file '%s' to fileset", filename)
	}

	// Calculate 0-based byte offset using helper.
	cursorPos, posErr := calculateCursorPos(file, row, col)
	if posErr != nil {
		return ctx, fmt.Errorf("cannot determine valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		return ctx, fmt.Errorf("%w: invalid cursor position calculated (Pos: %d)", ErrPositionConversion, cursorPos)
	}

	offset := file.Offset(cursorPos)
	// Clamp offset to content bounds.
	if offset < 0 {
		offset = 0
	}
	if offset > len(content) {
		offset = len(content)
	}

	ctx.Prefix = content[:offset]
	ctx.Suffix = content[offset:]

	// Extract the full line content.
	lineStartPos := file.LineStart(row) // Use 1-based row here
	if !lineStartPos.IsValid() {
		// Cycle 3: Use slog
		slog.Warn("Could not get start position for line", "line", row, "file", filename)
		return ctx, nil
	}
	startOffset := file.Offset(lineStartPos)

	lineEndOffset := file.Size()
	fileLineCount := file.LineCount()
	if row < fileLineCount {
		// Get start of *next* line to find end of current line
		if nextLineStartPos := file.LineStart(row + 1); nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	if startOffset >= 0 && lineEndOffset >= startOffset && lineEndOffset <= len(content) {
		lineContent := content[startOffset:lineEndOffset]
		// Trim trailing newline characters (\n or \r\n)
		lineContent = strings.TrimRight(lineContent, "\n")
		lineContent = strings.TrimRight(lineContent, "\r")
		ctx.FullLine = lineContent
	} else {
		// Cycle 3: Use slog
		slog.Warn("Could not extract full line content",
			"row", row,
			"startOffset", startOffset,
			"lineEndOffset", lineEndOffset,
			"contentLen", len(content),
			"file", filename)
	}
	return ctx, nil
}

// calculateCursorPos converts 1-based line/col to 0-based token.Pos offset.
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
		// Allow cursor at start of line after last line.
		if line == fileLineCount+1 && col == 1 {
			// File.Pos takes 0-based offset. Size is the offset *after* the last character.
			return file.Pos(file.Size()), nil
		}
		return token.NoPos, fmt.Errorf("%w: line number %d exceeds file line count %d", ErrPositionOutOfRange, line, fileLineCount)
	}

	// LineStart takes 1-based line number.
	lineStartPos := file.LineStart(line)
	if !lineStartPos.IsValid() {
		return token.NoPos, fmt.Errorf("%w: cannot get start offset for line %d in file '%s'", ErrPositionConversion, line, file.Name())
	}

	// Offset takes 0-based offset. lineStartPos is 0-based relative to file start.
	lineStartOffset := file.Offset(lineStartPos)
	// Calculate target offset (0-based) from 1-based column.
	cursorOffset := lineStartOffset + col - 1

	// Determine end offset of the line (exclusive for range, inclusive for position)
	lineEndOffset := file.Size() // Default to end of file for last line
	if line < fileLineCount {
		// Get start of *next* line
		if nextLineStartPos := file.LineStart(line + 1); nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	// Clamp offset to line boundaries (inclusive start, exclusive end for range, but need inclusive end for position)
	// The position should be *at most* the offset of the newline character or EOF.
	finalOffset := cursorOffset
	if cursorOffset < lineStartOffset {
		finalOffset = lineStartOffset
		// Cycle 3: Use slog
		slog.Warn("Column resulted in offset before line start. Clamping.", "col", col, "offset", cursorOffset, "line_start", lineStartOffset)
	}
	// Check against the start of the *next* line or EOF.
	if cursorOffset > lineEndOffset {
		finalOffset = lineEndOffset
		// Cycle 3: Use slog
		slog.Warn("Column resulted in offset beyond line end. Clamping.", "col", col, "offset", cursorOffset, "line_end", lineEndOffset)
	}

	// File.Pos takes 0-based offset.
	pos := file.Pos(finalOffset)
	if !pos.IsValid() {
		// Cycle 3: Use slog
		slog.Error("Clamped offset resulted in invalid token.Pos. Using line start.", "offset", finalOffset, "line_start_pos", lineStartPos)
		// Fallback to line start position if clamped offset is invalid
		return lineStartPos, fmt.Errorf("%w: failed to calculate valid token.Pos for offset %d", ErrPositionConversion, finalOffset)
	}
	return pos, nil
}

// ============================================================================
// Stream Processing Helpers (Used by Ollama Client)
// ============================================================================

// streamCompletion reads the Ollama stream response and writes it to w.
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer) error {
	defer r.Close()
	reader := bufio.NewReader(r)
	lineCount := 0
	for {
		select {
		case <-ctx.Done():
			// Cycle 3: Use slog
			slog.Warn("Context cancelled during streaming", "error", ctx.Err())
			return ctx.Err()
		default:
		} // Check context.

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if len(line) > 0 { // Process final partial line if any
					if procErr := processLine(line, w); procErr != nil {
						return procErr
					}
				}
				// Cycle 3: Use slog
				slog.Debug("Stream processing finished (EOF)", "lines_processed", lineCount)
				return nil
			} // Handle EOF.
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			} // Check context after read error.
		}
		lineCount++
		if procErr := processLine(line, w); procErr != nil {
			return procErr
		} // Process line.
	}
}

// processLine decodes a single line from the Ollama stream and writes the content.
func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil
	} // Ignore empty lines.
	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		// Cycle 3: Use slog (Debug level for non-JSON lines)
		slog.Debug("Ignoring non-JSON line from Ollama stream", "line", string(line))
		return nil
	} // Tolerate non-JSON lines.
	if resp.Error != "" {
		// Cycle 3: Use slog
		slog.Error("Ollama stream reported an error", "error", resp.Error)
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	} // Check for errors in stream.
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		// Cycle 3: Use slog
		slog.Error("Error writing stream chunk to output", "error", err)
		return fmt.Errorf("error writing to output: %w", err)
	} // Write content.
	return nil
}
