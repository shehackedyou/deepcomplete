package main // Or a utility sub-package for the LSP server

import (
	"bufio"
	"bytes"
	"fmt"
	"log" // For warnings
	"unicode/utf8"

	// Use the appropriate LSP protocol package for your chosen library
	// Example using gopls internal types (replace if using a different library)
	"golang.org/x/tools/internal/lsp/protocol"
	// If not using gopls types, define equivalent Position struct:
	// type Position struct { Line uint32; Character uint32 }
	// Consider using a dedicated, tested offset conversion library if one exists
	// E.g., check "github.com/sourcegraph/go-lsp/lspext" or similar
)

// lspPositionToBytePosition converts LSP 0-based line/character (UTF-16 code units)
// to Go 1-based line/column (bytes) and 0-based byte offset.
// It requires the current file content.
func lspPositionToBytePosition(content []byte, lspPos protocol.Position) (line, col, byteOffset int, err error) {
	if content == nil {
		return 0, 0, -1, fmt.Errorf("file content is nil")
	}
	// LSP Position uses uint32, cast safely
	targetLine := int(lspPos.Line)
	targetUTF16Char := int(lspPos.Character)

	if targetLine < 0 {
		return 0, 0, -1, fmt.Errorf("invalid LSP line: %d", targetLine)
	}
	if targetUTF16Char < 0 {
		return 0, 0, -1, fmt.Errorf("invalid LSP character offset: %d", targetUTF16Char)
	}

	currentLine := 0          // 0-based line counter
	currentByteOffset := 0    // 0-based byte offset from start of file
	lineStartByteOffset := -1 // Sentinel value

	scanner := bufio.NewScanner(bytes.NewReader(content))
	for scanner.Scan() {
		lineTextBytes := scanner.Bytes() // Bytes of the current line (excluding newline)
		lineLengthBytes := len(lineTextBytes)
		// Determine newline length (\n vs \r\n) - Assume \n for now for simplicity
		// A more robust solution would detect line endings.
		newlineLengthBytes := 1

		if currentLine == targetLine {
			lineStartByteOffset = currentByteOffset

			byteOffsetInLine, convErr := utf16OffsetToBytes(lineTextBytes, targetUTF16Char)
			if convErr != nil {
				// If conversion fails (e.g., offset out of bounds), clamp to end of line
				log.Printf("Warning: utf16OffsetToBytes failed (line %d, char %d): %v. Clamping to line end.", targetLine, targetUTF16Char, convErr)
				byteOffsetInLine = lineLengthBytes // Use byte length as max offset
			}

			// --- Calculate final results ---
			line = currentLine + 1     // 1-based line
			col = byteOffsetInLine + 1 // 1-based byte column
			byteOffset = lineStartByteOffset + byteOffsetInLine
			return line, col, byteOffset, nil // Success
		}

		currentByteOffset += lineLengthBytes + newlineLengthBytes // Move to start of next line
		currentLine++
	}

	// Handle case where the target line is the line *after* the last line with content
	if currentLine == targetLine {
		// Check if character offset is valid for an empty line after content
		if targetUTF16Char == 0 {
			lineStartByteOffset = currentByteOffset
			line = currentLine + 1
			col = 1                          // Start of the empty new line
			byteOffset = lineStartByteOffset // Offset is at the start of this line
			return line, col, byteOffset, nil
		} else {
			return 0, 0, -1, fmt.Errorf("invalid character offset %d on line %d (after last line with content)", targetUTF16Char, targetLine)
		}
	}

	if err := scanner.Err(); err != nil {
		return 0, 0, -1, fmt.Errorf("error scanning file content: %w", err)
	}

	// If loop finishes without finding targetLine
	return 0, 0, -1, fmt.Errorf("LSP line %d not found in file (total lines %d)", targetLine, currentLine)
}

// utf16OffsetToBytes converts a 0-based UTF-16 code unit offset within a byte slice
// containing UTF-8 text to a 0-based byte offset.
// IMPORTANT: This needs extensive testing with various Unicode characters and editor behaviors.
func utf16OffsetToBytes(line []byte, utf16Offset int) (int, error) {
	if utf16Offset < 0 {
		return 0, fmt.Errorf("invalid utf16Offset: %d", utf16Offset)
	}
	if utf16Offset == 0 {
		return 0, nil
	}

	byteOffset := 0
	currentUTF16Offset := 0

	for byteOffset < len(line) {
		if currentUTF16Offset >= utf16Offset {
			// Reached target offset before processing next rune
			break
		}

		r, size := utf8.DecodeRune(line[byteOffset:])
		if r == utf8.RuneError && size == 1 {
			return byteOffset, fmt.Errorf("invalid UTF-8 sequence at byte offset %d", byteOffset)
		}

		// Calculate UTF-16 units for the rune
		utf16Units := 1
		if r > 0xFFFF { // Needs surrogate pair
			utf16Units = 2
		}

		// Check if adding this rune *would exceed* the target offset
		if currentUTF16Offset+utf16Units > utf16Offset {
			// We landed *within* the UTF-16 units of this rune.
			// The LSP specification generally means the offset is *before* the character at that index.
			// So, the byte offset should be *before* this rune.
			break // Exit loop, current byteOffset is correct
		}

		// If it doesn't exceed, process the rune fully
		currentUTF16Offset += utf16Units
		byteOffset += size

		// Check if we exactly match the offset *after* processing the rune
		if currentUTF16Offset == utf16Offset {
			break
		}
	}

	// After loop, check if we reached the target offset or went past the line end
	if currentUTF16Offset < utf16Offset {
		// The target offset is beyond the actual length of the line in UTF-16 units
		// Clamp to the end of the line bytes
		return len(line), fmt.Errorf("utf16Offset %d is beyond the line length in UTF-16 units (%d)", utf16Offset, currentUTF16Offset)
	}

	// If we exited loop because currentUTF16Offset >= utf16Offset, byteOffset should be correct
	return byteOffset, nil
}
