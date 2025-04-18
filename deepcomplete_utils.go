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
	"log"
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
	fmt.Fprint(os.Stderr, color, text, ColorReset)
}

// ============================================================================
// LSP Position Conversion Helpers
// ============================================================================

// LSPPosition represents a 0-based line/character offset (UTF-16).
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
		newlineLengthBytes := 1
		if currentLine == targetLine {
			byteOffsetInLine, convErr := Utf16OffsetToBytes(lineTextBytes, targetUTF16Char)
			if convErr != nil {
				if errors.Is(convErr, ErrPositionOutOfRange) { // Clamp to line end on out-of-range error.
					log.Printf("Warning: utf16OffsetToBytes reported offset out of range (line %d, char %d): %v. Clamping to line end.", targetLine, targetUTF16Char, convErr)
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
		log.Printf("Warning: Error reading %s: %v", goModPath, err)
		return "read-error"
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		log.Printf("Warning: Error hashing %s: %v", goModPath, err)
		return "hash-error"
	}
	return hex.EncodeToString(h.Sum(nil))
}

// calculateInputHashes calculates hashes for go.mod, go.sum, and Go files.
// Uses pkg.CompiledGoFiles if available, otherwise scans the directory (non-recursive).
// Phase 2, Step 4: Added pkg argument.
func calculateInputHashes(dir string, pkg *packages.Package) (map[string]string, error) {
	hashes := make(map[string]string)
	filesToHash := make(map[string]struct{})

	// Always check for go.mod/go.sum.
	for _, fname := range []string{"go.mod", "go.sum"} {
		fpath := filepath.Join(dir, fname)
		if _, err := os.Stat(fpath); err == nil {
			if absPath, absErr := filepath.Abs(fpath); absErr == nil {
				filesToHash[absPath] = struct{}{}
			} else {
				log.Printf("Warning: Could not get absolute path for %s: %v", fpath, absErr)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("failed to stat %s: %w", fpath, err)
		}
	}

	// Phase 2, Step 4: Use compiled files from package info if available.
	filesFromPkg := false
	if pkg != nil && len(pkg.CompiledGoFiles) > 0 {
		filesFromPkg = true
		log.Printf("DEBUG: Hashing based on %d CompiledGoFiles from package %s", len(pkg.CompiledGoFiles), pkg.PkgPath)
		for _, fpath := range pkg.CompiledGoFiles {
			// CompiledGoFiles should already be absolute paths.
			filesToHash[fpath] = struct{}{}
		}
	}

	// Fallback: Scan directory for .go files if package info unavailable/empty.
	if !filesFromPkg {
		log.Printf("DEBUG: Calculating input hashes by scanning directory %s (pkg info unavailable or empty)", dir)
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, fmt.Errorf("failed to scan directory %s for hashing: %w", dir, err)
		}
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") {
				absPath := filepath.Join(dir, entry.Name())
				// No need for Abs here, Join already makes it absolute if dir is.
				filesToHash[absPath] = struct{}{}
			}
		}
	}

	// Calculate hash for each unique file path.
	for absPath := range filesToHash {
		relPath, err := filepath.Rel(dir, absPath)
		if err != nil {
			log.Printf("Warning: Could not get relative path for %s in %s: %v. Using base name.", absPath, dir, err)
			relPath = filepath.Base(absPath)
		}
		relPath = filepath.ToSlash(relPath) // Use forward slashes for consistent keys.

		hash, err := hashFileContent(absPath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				log.Printf("Warning: File %s disappeared during hashing. Skipping.", absPath)
				continue
			}
			return nil, fmt.Errorf("%w: failed to hash input file %s: %w", ErrCacheHash, absPath, err)
		}
		hashes[relPath] = hash
	}
	log.Printf("DEBUG: Calculated hashes for %d input files in %s", len(hashes), dir)
	return hashes, nil
}

// hashFileContent calculates the SHA256 hash of a single file.
func hashFileContent(filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return "", err
	} // Propagate open errors.
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	} // Propagate hashing errors.
	return hex.EncodeToString(h.Sum(nil)), nil
}

// compareFileHashes compares current and cached file hashes, logging differences.
func compareFileHashes(current, cached map[string]string) bool {
	if len(current) != len(cached) {
		log.Printf("DEBUG: Cache invalid: File count mismatch (Current: %d, Cached: %d)", len(current), len(cached))
		for relPath := range current {
			if _, ok := cached[relPath]; !ok {
				log.Printf("DEBUG:   - File '%s' exists now but was not in cache.", relPath)
			}
		}
		for relPath := range cached {
			if _, ok := current[relPath]; !ok {
				log.Printf("DEBUG:   - File '%s' was cached but does not exist now.", relPath)
			}
		}
		return false
	}
	for relPath, currentHash := range current {
		cachedHash, ok := cached[relPath]
		if !ok {
			log.Printf("DEBUG: Cache invalid: File '%s' missing in cache despite matching counts.", relPath)
			return false
		} // Should not happen if counts match.
		if currentHash != cachedHash {
			log.Printf("DEBUG: Cache invalid: Hash mismatch for file '%s'.", relPath)
			return false
		}
	}
	return true // Counts match and all hashes match.
}

// deleteCacheEntryByKey removes an entry directly using the key.
// Assumed to be called by the refactored InvalidateCache.
func deleteCacheEntryByKey(db *bbolt.DB, cacheKey []byte) error {
	if db == nil {
		return errors.New("cannot delete cache entry: db is nil")
	}
	err := db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(cacheBucketName)
		if b == nil {
			return nil
		} // Bucket gone? Nothing to do.
		if b.Get(cacheKey) == nil {
			return nil
		} // Key doesn't exist.
		log.Printf("DEBUG: Deleting cache entry for key: %s", string(cacheKey))
		return b.Delete(cacheKey)
	})
	if err != nil {
		log.Printf("Warning: Failed to delete cache entry %s: %v", string(cacheKey), err)
		return fmt.Errorf("%w: failed to delete entry %s: %w", ErrCacheWrite, string(cacheKey), err)
	}
	return nil
}

// ============================================================================
// Retry Helper
// ============================================================================

// retry executes an operation function with backoff and retry logic.
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration) error {
	var lastErr error
	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled before attempt %d: %v", i+1, ctx.Err())
			return ctx.Err()
		default:
		} // Check context.
		lastErr = operation()
		if lastErr == nil {
			return nil
		} // Success.
		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			log.Printf("Attempt %d failed due to context error: %v. Not retrying.", i+1, lastErr)
			return lastErr
		} // Don't retry context errors.
		var ollamaErr *OllamaError
		isRetryable := errors.As(lastErr, &ollamaErr) && (ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		isRetryable = isRetryable || errors.Is(lastErr, ErrOllamaUnavailable) // Check retryable errors.
		if !isRetryable {
			log.Printf("Attempt %d failed with non-retryable error: %v", i+1, lastErr)
			return lastErr
		} // Non-retryable.
		waitDuration := currentDelay
		log.Printf("Attempt %d failed with retryable error: %v. Retrying in %v...", i+1, lastErr, waitDuration)
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled during retry wait: %v", ctx.Err())
			return ctx.Err()
		case <-time.After(waitDuration): /* Optionally increase delay: currentDelay *= 2 */
		} // Wait or cancel.
	}
	log.Printf("Operation failed after %d retries.", maxRetries)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr) // Return last error after retries exhausted.
}

// ============================================================================
// Spinner
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
			log.Println("Warning: Timeout waiting for spinner goroutine cleanup")
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
	file := fset.AddFile(filename, 1, len(contentBytes)) // Base 1, correct size.
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
	lineStartPos := file.LineStart(row)
	if !lineStartPos.IsValid() {
		log.Printf("Warning: Could not get start position for line %d in %s", row, filename)
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
		// Trim trailing newline characters.
		lineContent = strings.TrimRight(lineContent, "\n")
		lineContent = strings.TrimRight(lineContent, "\r")
		ctx.FullLine = lineContent
	} else {
		log.Printf("Warning: Could not extract full line for row %d (start %d, end %d, content len %d) in %s", row, startOffset, lineEndOffset, len(content), filename)
	}
	return ctx, nil
}

// calculateCursorPos converts 1-based line/col to 0-based token.Pos offset.
// Moved here from deepcomplete.go as it doesn't depend on DeepCompleter state.
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
		} // Allow cursor at start of line after last line.
		return token.NoPos, fmt.Errorf("%w: line number %d exceeds file line count %d", ErrPositionOutOfRange, line, fileLineCount)
	}

	lineStartPos := file.LineStart(line)
	if !lineStartPos.IsValid() {
		return token.NoPos, fmt.Errorf("%w: cannot get start offset for line %d in file '%s'", ErrPositionConversion, line, file.Name())
	}

	lineStartOffset := file.Offset(lineStartPos)
	cursorOffset := lineStartOffset + col - 1 // Calculate target offset.

	// Determine end offset of the line.
	lineEndOffset := file.Size()
	if line < fileLineCount {
		if nextLineStartPos := file.LineStart(line + 1); nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	// Clamp offset to line boundaries.
	finalOffset := cursorOffset
	if cursorOffset < lineStartOffset {
		finalOffset = lineStartOffset
		log.Printf("Warning: column %d resulted in offset %d before line start %d. Clamping.", col, cursorOffset, lineStartOffset)
	}
	if cursorOffset > lineEndOffset {
		finalOffset = lineEndOffset
		log.Printf("Warning: column %d resulted in offset %d beyond line end %d. Clamping.", col, cursorOffset, lineEndOffset)
	}

	pos := file.Pos(finalOffset)
	if !pos.IsValid() {
		log.Printf("Error: Clamped offset %d resulted in invalid token.Pos. Using line start %d.", finalOffset, lineStartPos)
		return lineStartPos, fmt.Errorf("%w: failed to calculate valid token.Pos for offset %d", ErrPositionConversion, finalOffset)
	}
	return pos, nil
}

// ============================================================================
// Stream Processing Helpers (Used by Ollama Client)
// ============================================================================

// streamCompletion reads the Ollama stream response and writes it to w.
// Moved here from deepcomplete.go
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer) error {
	defer r.Close()
	reader := bufio.NewReader(r)
	for {
		select {
		case <-ctx.Done():
			log.Println("Context cancelled during streaming")
			return ctx.Err()
		default:
		} // Check context.
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if len(line) > 0 {
					if procErr := processLine(line, w); procErr != nil {
						return procErr
					}
				}
				return nil
			} // Handle EOF.
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			} // Check context after read error.
		}
		if procErr := processLine(line, w); procErr != nil {
			return procErr
		} // Process line.
	}
}

// processLine decodes a single line from the Ollama stream and writes the content.
// Moved here from deepcomplete.go
func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil
	} // Ignore empty lines.
	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		log.Printf("Debug: Ignoring non-JSON line from Ollama stream: %s", string(line))
		return nil
	} // Tolerate non-JSON lines.
	if resp.Error != "" {
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	} // Check for errors in stream.
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		return fmt.Errorf("error writing to output: %w", err)
	} // Write content.
	return nil
}
