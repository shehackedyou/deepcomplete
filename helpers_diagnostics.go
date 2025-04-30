// deepcomplete/helpers_diagnostics.go
package deepcomplete

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log/slog"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
)

// ============================================================================
// Diagnostic Helpers
// ============================================================================

// addAnalysisError adds a non-fatal error to the info struct's list, avoiding duplicates.
func addAnalysisError(info *AstContextInfo, err error, logger *slog.Logger) {
	if err == nil || info == nil {
		return // No error or nowhere to add it
	}
	errMsg := err.Error()
	// Check for duplicates before adding
	for _, existing := range info.AnalysisErrors {
		if existing.Error() == errMsg {
			return // Avoid duplicate error messages
		}
	}
	logger.Warn("Analysis warning", "error", err) // Log the warning
	info.AnalysisErrors = append(info.AnalysisErrors, err)
}

// logAnalysisErrors logs joined non-fatal analysis errors if any occurred.
// Useful for summarizing errors at the end of an operation.
func logAnalysisErrors(errs []error, logger *slog.Logger) {
	if len(errs) > 0 {
		combinedErr := errors.Join(errs...) // Use errors.Join for cleaner wrapping
		logger.Warn("Context analysis completed with non-fatal errors", "count", len(errs), "errors", combinedErr)
	}
}

// packagesErrorToDiagnostic converts a packages.Error into our internal Diagnostic format.
// It parses the error position string and maps the error kind to severity.
func packagesErrorToDiagnostic(pkgErr packages.Error, fset *token.FileSet, logger *slog.Logger) *Diagnostic {
	if fset == nil {
		logger.Warn("Cannot convert packages.Error to Diagnostic: FileSet is nil")
		return nil
	}

	posStr := pkgErr.Pos // Format: "filename:line:col: message" or "filename:line:col"
	msg := pkgErr.Msg
	kind := pkgErr.Kind // 0=Unknown, 1=ParseError, 2=TypeError

	// Default position if parsing fails (start of file)
	startPos := Position{Line: 0, Character: 0} // 0-based line, 0-based byte offset
	endPos := Position{Line: 0, Character: 1}   // Default to 1 character length

	parts := strings.SplitN(posStr, ":", 4)
	var filename string
	var lineNum, colNum int = 1, 1 // 1-based defaults
	var parseErrs []error

	if len(parts) >= 3 {
		filename = parts[0]
		var lineErr, colErr error
		lineNum, lineErr = strconv.Atoi(parts[1])
		colNum, colErr = strconv.Atoi(parts[2])
		if lineErr != nil {
			parseErrs = append(parseErrs, fmt.Errorf("parsing line: %w", lineErr))
		}
		if colErr != nil {
			parseErrs = append(parseErrs, fmt.Errorf("parsing column: %w", colErr))
		}

		if len(parseErrs) == 0 {
			// Successfully parsed line and column (1-based)
			// Find the corresponding token.File
			var tokenFile *token.File
			fset.Iterate(func(f *token.File) bool {
				// Note: Comparing file names might be fragile if paths differ (abs vs rel).
				// Consider normalizing paths before comparison if issues arise.
				// For now, assume names provided by packages.Load are consistent within the fset.
				if f != nil && f.Name() == filename {
					tokenFile = f
					return false // Stop iteration
				}
				return true // Continue iteration
			})

			if tokenFile != nil {
				// Convert 1-based line/col to 0-based byte offset
				lineStartPos := tokenFile.LineStart(lineNum)
				if lineStartPos.IsValid() {
					lineStartOffset := tokenFile.Offset(lineStartPos)
					// Calculate 0-based byte offset from file start
					// Note: packages.Error 'col' is often byte-based, but treat as character for safety.
					// A more robust approach might involve reading the line content.
					// For simplicity, assume col is byte-based for now.
					byteOffset := lineStartOffset + colNum - 1

					// Ensure offset is within file bounds
					if byteOffset < 0 {
						byteOffset = 0
					}
					if byteOffset > tokenFile.Size() {
						byteOffset = tokenFile.Size()
					}

					// Set diagnostic start position (0-based line, 0-based byte char)
					startPos.Line = lineNum - 1
					startPos.Character = byteOffset // Use byte offset directly for internal format

					// Default end position to start + 1 byte for now
					endPos = startPos
					endPos.Character++
					// Ensure end position doesn't exceed file size
					if endPos.Character > tokenFile.Size() {
						endPos.Character = tokenFile.Size()
					}

				} else {
					parseErrs = append(parseErrs, fmt.Errorf("could not find start position for line %d", lineNum))
				}
			} else {
				parseErrs = append(parseErrs, fmt.Errorf("file '%s' not found in FileSet", filename))
			}
		}

		// Refine message if it was part of the position string
		if len(parts) == 4 && len(parseErrs) == 0 { // Only use message part if pos parsed ok
			msg = strings.TrimSpace(parts[3])
		}
	} else if posStr != "" { // Position string exists but couldn't be parsed
		parseErrs = append(parseErrs, errors.New("position string format not filename:line:col"))
	}

	// Log parsing errors if any occurred
	if len(parseErrs) > 0 {
		logger.Warn("Failed to parse position from packages.Error, using default range", "pos_string", posStr, "errors", errors.Join(parseErrs...))
		// Reset to default range if parsing failed
		startPos = Position{Line: 0, Character: 0}
		endPos = Position{Line: 0, Character: 1}
	}

	// Determine severity
	severity := SeverityError // Default to error
	if kind == packages.TypeError {
		severity = SeverityError
	} else if kind == packages.ParseError {
		severity = SeverityError
	}

	return &Diagnostic{
		Range:    Range{Start: startPos, End: endPos},
		Severity: severity,
		Source:   "go", // Source is the Go compiler/type checker
		Message:  msg,
		// Code: Could potentially extract common error codes from msg later
	}
}

// createDiagnosticForNode creates a diagnostic targeting a specific AST node's range.
func createDiagnosticForNode(fset *token.FileSet, node ast.Node, severity DiagnosticSeverity, code, source, message string, logger *slog.Logger) *Diagnostic {
	if fset == nil || node == nil {
		logger.Warn("Cannot create diagnostic for node: FileSet or Node is nil")
		return nil
	}

	startTokenPos := node.Pos()
	endTokenPos := node.End()

	if !startTokenPos.IsValid() || !endTokenPos.IsValid() {
		logger.Warn("Cannot create diagnostic: Node position is invalid", "node_type", fmt.Sprintf("%T", node))
		return nil
	}

	file := fset.File(startTokenPos)
	if file == nil {
		logger.Warn("Cannot create diagnostic: Could not get token.File for node", "node_type", fmt.Sprintf("%T", node))
		return nil
	}

	startOffset := file.Offset(startTokenPos)
	endOffset := file.Offset(endTokenPos)

	// Validate offsets
	if startOffset < 0 || endOffset < 0 || endOffset < startOffset || endOffset > file.Size() || startOffset > file.Size() {
		logger.Warn("Cannot create diagnostic: Invalid offsets calculated from node", "node_type", fmt.Sprintf("%T", node), "start", startOffset, "end", endOffset, "file_size", file.Size())
		// Attempt to recover with a zero-length range at the start if possible
		if startOffset >= 0 && startOffset <= file.Size() {
			endOffset = startOffset
		} else {
			return nil // Cannot recover
		}
	}

	// Get 0-based line numbers corresponding to the offsets
	startLineNum := file.Line(startTokenPos) - 1
	endLineNum := file.Line(endTokenPos) - 1
	if startLineNum < 0 || endLineNum < 0 {
		logger.Warn("Cannot create diagnostic: Invalid line numbers calculated", "start_line", startLineNum+1, "end_line", endLineNum+1)
		return nil
	}

	// Create internal Diagnostic Range using 0-based line and 0-based byte offsets
	startDiagPos := Position{
		Line:      startLineNum,
		Character: startOffset, // Use direct byte offset for internal representation
	}
	endDiagPos := Position{
		Line:      endLineNum,
		Character: endOffset, // Use direct byte offset for internal representation
	}

	return &Diagnostic{
		Range:    Range{Start: startDiagPos, End: endDiagPos},
		Severity: severity,
		Code:     code,
		Source:   source,
		Message:  message,
	}
}

// addAnalysisDiagnostics generates diagnostics based on common analysis findings.
func addAnalysisDiagnostics(fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	// Check for unresolved identifiers found during context node identification
	if info.IdentifierAtCursor != nil && info.IdentifierObject == nil && info.IdentifierType == nil {
		// Check if it's a known builtin before reporting as unresolved
		isBuiltin := false
		if obj, _ := types.Universe.Lookup(info.IdentifierAtCursor.Name).(*types.Builtin); obj != nil {
			isBuiltin = true
		}
		if !isBuiltin {
			diag := createDiagnosticForNode(
				fset,
				info.IdentifierAtCursor,
				SeverityError, // Unresolved identifier is typically an error
				"unresolved-identifier",
				"deepcomplete-analyzer",
				fmt.Sprintf("unresolved identifier: %s", info.IdentifierAtCursor.Name),
				logger,
			)
			if diag != nil {
				info.Diagnostics = append(info.Diagnostics, *diag)
			}
		}
	}

	// Check for unknown selector
	if info.SelectorExpr != nil && info.SelectorExprType != nil && info.SelectorExpr.Sel != nil {
		selName := info.SelectorExpr.Sel.Name
		// Use LookupFieldOrMethod to check validity
		// Check both value and pointer receivers (true for pointer)
		obj, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
		if obj == nil {
			// Check if it's a method on a pointer receiver type, but we have the non-pointer type
			// (This is implicitly handled by LookupFieldOrMethod with pointer=true)

			diag := createDiagnosticForNode(
				fset,
				info.SelectorExpr.Sel, // Diagnose the selector identifier itself
				SeverityError,         // Unknown member is an error
				"unknown-member",
				"deepcomplete-analyzer",
				fmt.Sprintf("unknown member '%s' for type '%s'", selName, info.SelectorExprType.String()),
				logger,
			)
			if diag != nil {
				info.Diagnostics = append(info.Diagnostics, *diag)
			}
		}
	}
	// TODO: Add more checks: type mismatches in assignments/calls, unused variables (requires more state tracking)
}

// getMissingTypeInfoReason provides a string explaining why type info might be missing.
func getMissingTypeInfoReason(targetPkg *packages.Package) string {
	if targetPkg == nil {
		return "target package is nil"
	}
	if targetPkg.TypesInfo == nil {
		return "TypesInfo is nil"
	}
	if targetPkg.TypesInfo.Defs == nil {
		return "TypesInfo.Defs is nil"
	}
	if targetPkg.TypesInfo.Scopes == nil {
		return "TypesInfo.Scopes is nil"
	}
	if targetPkg.TypesInfo.Types == nil {
		return "TypesInfo.Types is nil"
	}
	return "reason unknown (package and TypesInfo fields seem present)"
}

// getPosString safely gets a position string or returns a placeholder.
func getPosString(fset *token.FileSet, pos token.Pos) string {
	if fset != nil && pos.IsValid() {
		return fset.Position(pos).String()
	}
	return fmt.Sprintf("Pos(%d)", pos)
}
