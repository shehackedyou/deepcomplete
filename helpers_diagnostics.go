// deepcomplete/helpers_diagnostics.go
// Contains helper functions specifically for creating and managing diagnostics.
package deepcomplete

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log/slog"
	"path/filepath" // Added for path normalization
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
		return
	}
	if logger == nil {
		logger = slog.Default() // Fallback
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
func logAnalysisErrors(errs []error, logger *slog.Logger) {
	if len(errs) > 0 {
		if logger == nil {
			logger = slog.Default() // Fallback
		}
		combinedErr := errors.Join(errs...) // Use errors.Join for cleaner wrapping
		logger.Warn("Context analysis completed with non-fatal errors", "count", len(errs), "errors", combinedErr)
	}
}

// packagesErrorToDiagnostic converts a packages.Error into our internal Diagnostic format.
// Tries to parse position for a more accurate range.
func packagesErrorToDiagnostic(pkgErr packages.Error, fset *token.FileSet, logger *slog.Logger) *Diagnostic {
	if logger == nil {
		logger = slog.Default() // Fallback
	}
	if fset == nil {
		logger.Warn("Cannot convert packages.Error to Diagnostic: FileSet is nil")
		return nil
	}

	posStr := pkgErr.Pos // Format: "filename:line:col: message" or "filename:line:col" or just "filename:"
	msg := pkgErr.Msg
	kind := pkgErr.Kind // 0=Unknown, 1=ParseError, 2=TypeError

	// Default position (start of file) and range
	startPos := Position{Line: 0, Character: 0}
	endPos := Position{Line: 0, Character: 1} // Default 1 byte length

	parts := strings.SplitN(posStr, ":", 4)
	var filename string
	var lineNum, colNum int = 1, 1 // 1-based defaults
	var parseErrs []error

	if len(parts) >= 1 && parts[0] != "" {
		filename = parts[0]
		absFilename, absErr := filepath.Abs(filename)
		if absErr != nil {
			logger.Warn("Could not get absolute path for diagnostic filename", "filename", filename, "error", absErr)
		} else {
			filename = absFilename
		}
	} else {
		parseErrs = append(parseErrs, errors.New("cannot determine filename from position string"))
		filename = "[unknown]"
	}

	if len(parts) >= 3 {
		var lineErr, colErr error
		lineNum, lineErr = strconv.Atoi(parts[1])
		colNum, colErr = strconv.Atoi(parts[2])
		if lineErr != nil {
			parseErrs = append(parseErrs, fmt.Errorf("parsing line: %w", lineErr))
		}
		if colErr != nil {
			parseErrs = append(parseErrs, fmt.Errorf("parsing column: %w", colErr))
		}
		if lineNum <= 0 {
			lineNum = 1
			parseErrs = append(parseErrs, errors.New("line number is not positive"))
		}
		if colNum <= 0 {
			colNum = 1
			parseErrs = append(parseErrs, errors.New("column number is not positive"))
		}

		if len(parseErrs) == 0 {
			var tokenFile *token.File
			fset.Iterate(func(f *token.File) bool {
				if f != nil {
					fAbs, fAbsErr := filepath.Abs(f.Name())
					if fAbsErr == nil && fAbs == filename {
						tokenFile = f
						return false
					}
					if fAbsErr != nil && filepath.Base(f.Name()) == filepath.Base(filename) {
						logger.Debug("Matched diagnostic file by base name (absolute path failed/mismatched)", "f_name", f.Name(), "diag_filename", filename)
						tokenFile = f
						return false
					}
				}
				return true
			})

			if tokenFile != nil {
				// Convert 1-based line/col to 0-based byte offset
				tokenPos, calcErr := calculateCursorPos(tokenFile, lineNum, colNum, logger)
				if calcErr == nil && tokenPos.IsValid() {
					byteOffset := tokenFile.Offset(tokenPos)
					startPos.Line = lineNum - 1
					startPos.Character = byteOffset // Use byte offset for internal range start

					endPos = startPos
					if byteOffset < tokenFile.Size() {
						endPos.Character++ // Default to 1 byte length
					}

				} else {
					parseErrs = append(parseErrs, fmt.Errorf("could not calculate valid position for line %d, col %d: %w", lineNum, colNum, calcErr))
				}
			} else {
				parseErrs = append(parseErrs, fmt.Errorf("file '%s' not found in FileSet", filename))
			}
		}

		// Refine message if it was part of the position string
		if len(parts) == 4 && len(parseErrs) == 0 {
			msg = strings.TrimSpace(parts[3])
		}
	} else if posStr != "" && !strings.HasSuffix(posStr, ":") {
		parseErrs = append(parseErrs, errors.New("position string format not filename:line:col"))
	}

	if len(parseErrs) > 0 {
		logger.Warn("Failed to parse position from packages.Error, using default range", "pos_string", posStr, "errors", errors.Join(parseErrs...))
		startPos = Position{Line: 0, Character: 0}
		endPos = Position{Line: 0, Character: 1}
	}

	severity := SeverityError // Default to error
	if kind == packages.TypeError || kind == packages.ParseError {
		severity = SeverityError
	} else if strings.Contains(strings.ToLower(msg), "warning") { // Basic check for warnings
		severity = SeverityWarning
	}

	return &Diagnostic{
		Range:    Range{Start: startPos, End: endPos},
		Severity: severity,
		Source:   "go", // Source is the Go compiler/type checker
		Message:  msg,
	}
}

// createDiagnosticForNode creates a diagnostic targeting a specific AST node's range.
// Uses the node's Pos() and End() for the range.
func createDiagnosticForNode(fset *token.FileSet, node ast.Node, severity DiagnosticSeverity, code, source, message string, logger *slog.Logger) *Diagnostic {
	if logger == nil {
		logger = slog.Default() // Fallback
	}
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
		logger.Warn("Cannot create diagnostic: Could not get token.File for node", "node_type", fmt.Sprintf("%T", node), "start_pos", startTokenPos)
		return nil
	}
	if fset.File(endTokenPos) != file {
		logger.Warn("Cannot create diagnostic: Node spans multiple files", "node_type", fmt.Sprintf("%T", node), "start_file", file.Name(), "end_file", fset.File(endTokenPos).Name())
		return nil
	}

	startOffset := file.Offset(startTokenPos)
	endOffset := file.Offset(endTokenPos)
	fileSize := file.Size()

	// Validate offsets
	if startOffset < 0 || endOffset < 0 || startOffset > fileSize || endOffset > fileSize || endOffset < startOffset {
		logger.Warn("Cannot create diagnostic: Invalid offsets calculated from node", "node_type", fmt.Sprintf("%T", node), "start", startOffset, "end", endOffset, "file_size", fileSize)
		if startOffset >= 0 && startOffset <= fileSize {
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
		Character: startOffset, // Use direct byte offset
	}
	endDiagPos := Position{
		Line:      endLineNum,
		Character: endOffset, // Use direct byte offset
	}

	// Ensure end is not before start (can happen with empty nodes)
	if endDiagPos.Line < startDiagPos.Line || (endDiagPos.Line == startDiagPos.Line && endDiagPos.Character < startDiagPos.Character) {
		endDiagPos = startDiagPos
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
	if logger == nil {
		logger = slog.Default() // Fallback
	}
	if info == nil || fset == nil {
		logger.Warn("Cannot add analysis diagnostics: AstContextInfo or FileSet is nil")
		return
	}

	// Check for unresolved identifiers found during context node identification
	if info.IdentifierAtCursor != nil && info.IdentifierObject == nil && info.IdentifierType == nil {
		isBuiltin := false
		if obj, ok := types.Universe.Lookup(info.IdentifierAtCursor.Name).(*types.Builtin); ok && obj != nil {
			isBuiltin = true
		}
		if !isBuiltin {
			diag := createDiagnosticForNode( // Pass logger
				fset,
				info.IdentifierAtCursor,
				SeverityError,
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
		obj, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
		if obj == nil {
			diag := createDiagnosticForNode( // Pass logger
				fset,
				info.SelectorExpr.Sel,
				SeverityError,
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

	// Basic check for potentially unused variables (very naive)
	// Use info.TargetPackage instead of info.TargetPkg
	if info.TargetPackage != nil && info.TargetPackage.TypesInfo != nil && info.TargetPackage.TypesInfo.Defs != nil {
		for ident, obj := range info.TargetPackage.TypesInfo.Defs {
			// Check only for variables defined within the current function scope
			// Use info.TargetPackage instead of info.TargetPkg
			if v, ok := obj.(*types.Var); ok && !v.IsField() && v.Parent() != nil && v.Parent() != info.TargetPackage.Types.Scope() {
				// Check if the variable is used anywhere (simple check)
				// Use info.TargetPackage instead of info.TargetPkg
				if _, used := info.TargetPackage.TypesInfo.Uses[ident]; !used {
					if ident.Name == "_" {
						continue
					}
					diag := createDiagnosticForNode(
						fset,
						ident,
						SeverityHint, // Unused variable is often just a hint/warning
						"unused-variable",
						"deepcomplete-analyzer",
						fmt.Sprintf("variable '%s' declared but not used (basic check)", ident.Name),
						logger,
					)
					if diag != nil {
						isDuplicate := false
						for _, existingDiag := range info.Diagnostics {
							if existingDiag.Code == "unused-variable" && strings.Contains(existingDiag.Message, fmt.Sprintf("'%s'", ident.Name)) {
								isDuplicate = true
								break
							}
						}
						if !isDuplicate {
							info.Diagnostics = append(info.Diagnostics, *diag)
						}
					}
				}
			}
		}
	}

	// TODO: Add more checks: type mismatches in assignments/calls etc.
}

// getMissingTypeInfoReason provides a string explaining why type info might be missing.
func getMissingTypeInfoReason(targetPkg *packages.Package) string {
	if targetPkg == nil {
		return "target package is nil"
	}
	if targetPkg.TypesInfo == nil {
		return "TypesInfo is nil"
	}
	return "reason unknown (package and TypesInfo fields seem present)"
}

// getPosString is defined in deepcomplete_utils.go
