// deepcomplete_helpers.go
// Contains internal helper functions moved from deepcomplete.go
// as part of the analysis modularization (Cycle 8).
package deepcomplete

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"log/slog" // Use slog
	"path/filepath"
	"sort"
	"strings"
	"time" // Cycle 9: Needed for TTL

	// Cycle 9: Added ristretto import (needed for cache checks)
	// "github.com/dgraph-io/ristretto"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

// ** ADDED: loadPackageAndFile (Moved from original Analyze logic) **
// loadPackageAndFile encapsulates the logic for loading package information.
func loadPackageAndFile(ctx context.Context, absFilename string, fset *token.FileSet, logger *slog.Logger) (*packages.Package, *ast.File, *token.File, []error) {
	var loadErrors []error
	dir := filepath.Dir(absFilename)
	logger = logger.With("loadDir", dir) // Add specific context for this function

	// Configuration for loading packages. NeedSyntax is crucial for AST.
	loadCfg := &packages.Config{
		Context: ctx,
		Dir:     dir,
		Fset:    fset,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedTypesSizes |
			packages.NeedSyntax | packages.NeedTypesInfo,
		// Load imports recursively? Might be too slow/memory intensive initially.
		// Mode: packages.LoadAllSyntax, // Alternative: Load syntax for all dependencies
		Tests: false,                                                                                   // Don't load test files initially
		Logf:  func(format string, args ...interface{}) { logger.Debug(fmt.Sprintf(format, args...)) }, // Pipe packages logs to slog
	}

	logger.Debug("Calling packages.Load")
	pkgs, err := packages.Load(loadCfg, fmt.Sprintf("file=%s", absFilename))
	if err != nil {
		// This is usually a critical error (e.g., context cancelled, setup issue)
		loadErrors = append(loadErrors, fmt.Errorf("packages.Load failed critically: %w", err))
		logger.Error("packages.Load failed critically", "error", err)
		return nil, nil, nil, loadErrors
	}

	if len(pkgs) == 0 {
		loadErrors = append(loadErrors, errors.New("packages.Load returned no packages"))
		logger.Warn("packages.Load returned no packages.")
		return nil, nil, nil, loadErrors
	}

	var targetPkg *packages.Package
	var targetFileAST *ast.File
	var targetFile *token.File

	// Collect all package-level errors first
	for _, p := range pkgs {
		for i := range p.Errors {
			// Make a copy of the error to avoid loop variable capture issues if used in goroutines later
			loadErr := p.Errors[i]
			loadErrors = append(loadErrors, &loadErr) // Store as pointer to original error type
			logger.Warn("Package loading error encountered", "package", p.PkgPath, "error", loadErr)
		}
	}

	// Find the specific package and AST file corresponding to absFilename
	for _, p := range pkgs {
		if p == nil {
			continue
		}
		logger.Debug("Checking package", "pkg_id", p.ID, "pkg_path", p.PkgPath, "num_files", len(p.CompiledGoFiles), "num_syntax", len(p.Syntax))
		for i, astFile := range p.Syntax {
			if astFile == nil {
				logger.Warn("Nil AST file encountered in package syntax", "package", p.PkgPath, "index", i)
				continue
			}
			// Use the FileSet to get the filename associated with the AST node's position
			filePos := fset.Position(astFile.Pos()) // Get position from AST node
			if !filePos.IsValid() {
				logger.Warn("AST file has invalid position", "package", p.PkgPath, "index", i)
				continue
			}
			// Compare absolute paths for robustness
			astFilePath, _ := filepath.Abs(filePos.Filename)
			logger.Debug("Comparing file paths", "ast_file_path", astFilePath, "target_file_path", absFilename)
			if astFilePath == absFilename {
				targetPkg = p
				targetFileAST = astFile
				// Get the token.File using the valid position
				targetFile = fset.File(astFile.Pos())
				if targetFile == nil {
					// This should ideally not happen if astFile.Pos() is valid
					loadErrors = append(loadErrors, fmt.Errorf("failed to get token.File for AST file %s", absFilename))
					logger.Error("Could not get token.File from FileSet even with valid AST position", "filename", absFilename)
				}
				logger.Debug("Found target file in package", "package", p.PkgPath)
				goto foundTarget // Exit loops once target is found
			}
		}
	}

foundTarget:
	// Handle case where the specific file wasn't found in the loaded packages' syntax trees
	if targetPkg == nil {
		err := fmt.Errorf("target file %s not found in loaded packages syntax trees", absFilename)
		loadErrors = append(loadErrors, err)
		logger.Warn(err.Error())
		// Maybe fallback to the first package if only one was loaded? Risky.
		// if len(pkgs) == 1 {
		//     targetPkg = pkgs[0]
		//     logger.Warn("Falling back to first loaded package, but target file AST/Token info will be missing.", "package", targetPkg.PkgPath)
		// }
	}

	// Final checks on the found package (if any)
	if targetPkg != nil {
		// Check if critical type information is missing, which indicates deeper issues
		if targetPkg.TypesInfo == nil {
			loadErrors = append(loadErrors, fmt.Errorf("type info (TypesInfo) is nil for target package %s", targetPkg.PkgPath))
			logger.Warn("TypesInfo is nil for target package", "package", targetPkg.PkgPath)
		}
		if targetPkg.Types == nil {
			loadErrors = append(loadErrors, fmt.Errorf("types (Types) is nil for target package %s", targetPkg.PkgPath))
			logger.Warn("Types is nil for target package", "package", targetPkg.PkgPath)
		}
	}

	return targetPkg, targetFileAST, targetFile, loadErrors
}

// ============================================================================
// Analysis Step Orchestration & Helpers (Cycle 8 Refactor)
// ============================================================================

// performAnalysisSteps orchestrates the detailed analysis after loading.
// Calls modular helper functions, potentially wrapped by memory cache (Cycle 9).
// Expects absFilename in info to be validated.
func performAnalysisSteps(
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package,
	fset *token.FileSet,
	line, col int,
	analyzer *GoPackagesAnalyzer, // Cycle 9: Pass analyzer for memory cache access
	info *AstContextInfo, // Pass info struct to be populated
	logger *slog.Logger,
) error {
	// Defensive check for required inputs
	if targetFile == nil || fset == nil {
		err := errors.New("performAnalysisSteps requires non-nil targetFile and fset")
		addAnalysisError(info, err, logger)
		return err // Cannot proceed without file/fileset
	}

	// Calculate cursor position first
	cursorPos, posErr := calculateCursorPos(targetFile, line, col) // Uses utils helper now
	if posErr != nil {
		// If position is invalid, we cannot proceed with AST-based analysis
		err := fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
		addAnalysisError(info, err, logger)
		// Attempt to gather package scope anyway, as it doesn't depend on cursor position
		gatherScopeContext(nil, targetPkg, fset, info, logger) // Pass nil path
		return err                                             // Return error to indicate failure
	}
	info.CursorPos = cursorPos // Store valid cursor position
	logger = logger.With("cursorPos", info.CursorPos, "cursorPosStr", fset.PositionFor(info.CursorPos, true).String())
	logger.Debug("Calculated cursor position")

	// Proceed only if AST is available
	if targetFileAST != nil {
		// --- Call modular analysis functions ---
		// These functions might now use the memory cache via the analyzer instance.

		// 1. Find enclosing path and identify specific context nodes (call, selector, etc.)
		// This might be cacheable based on file version + cursor position.
		// Conceptual: Wrap with memory cache check
		path, pathErr := findEnclosingPathAndNodes(targetFileAST, info.CursorPos, targetPkg, fset, analyzer, info, logger)
		if pathErr != nil {
			addAnalysisError(info, fmt.Errorf("failed to find enclosing path/nodes: %w", pathErr), logger)
			// Continue analysis if possible, but context might be less accurate
		}

		// 2. Extract scope information based on path and package info
		// This might be cacheable based on file version + block/function scope identifier.
		// Conceptual: Wrap with memory cache check
		scopeErr := extractScopeInformation(path, targetPkg, info.CursorPos, analyzer, info, logger)
		if scopeErr != nil {
			addAnalysisError(info, fmt.Errorf("failed to extract scope info: %w", scopeErr), logger)
		}

		// 3. Extract relevant comments near the cursor
		// This might be cacheable based on file version + cursor position/enclosing node.
		// Conceptual: Wrap with memory cache check
		commentErr := extractRelevantComments(targetFileAST, path, info.CursorPos, fset, analyzer, info, logger)
		if commentErr != nil {
			addAnalysisError(info, fmt.Errorf("failed to extract comments: %w", commentErr), logger)
		}

	} else {
		// AST is missing, likely due to load errors.
		addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil"), logger)
		// Attempt to gather package scope anyway
		gatherScopeContext(nil, targetPkg, fset, info, logger) // Pass nil path
	}

	// Return nil, non-fatal errors are collected in info.AnalysisErrors
	return nil
}

// ============================================================================
// Specific Analysis Step Implementations (Cycle 8 Refactor)
// ============================================================================

// findEnclosingPathAndNodes finds the AST path and identifies context nodes.
// Conceptual: Could be wrapped by memory cache.
func findEnclosingPathAndNodes(
	targetFileAST *ast.File,
	cursorPos token.Pos,
	pkg *packages.Package,
	fset *token.FileSet,
	analyzer *GoPackagesAnalyzer, // For potential caching
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) ([]ast.Node, error) {
	// --- Memory Cache Check (Conceptual) ---
	// ... (cache logic omitted for brevity) ...
	// --- End Cache Check ---

	// --- Direct Implementation (without cache wrapping for now) ---
	path := findEnclosingPath(targetFileAST, cursorPos, info, logger)
	// findContextNodes populates info directly and adds errors to info.AnalysisErrors
	findContextNodes(path, cursorPos, pkg, fset, analyzer, info, logger)
	// Return the path, errors are collected in info.AnalysisErrors
	return path, nil
}

// extractScopeInformation gathers variables/types in scope.
// Conceptual: Could be wrapped by memory cache.
func extractScopeInformation(
	path []ast.Node,
	targetPkg *packages.Package,
	cursorPos token.Pos,
	analyzer *GoPackagesAnalyzer, // For potential caching
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) error {
	// --- Memory Cache Check (Conceptual) ---
	// ... (cache logic omitted for brevity) ...
	// --- End Cache Check ---

	// --- Direct Implementation ---
	gatherScopeContext(path, targetPkg, info.TargetFileSet, info, logger) // Populates info
	return nil                                                            // Errors are added to info.AnalysisErrors internally
}

// extractRelevantComments finds comments near the cursor.
// Conceptual: Could be wrapped by memory cache.
func extractRelevantComments(
	targetFileAST *ast.File,
	path []ast.Node,
	cursorPos token.Pos,
	fset *token.FileSet,
	analyzer *GoPackagesAnalyzer, // For potential caching
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) error {
	// --- Memory Cache Check (Conceptual) ---
	// ... (cache logic omitted for brevity) ...
	// --- End Cache Check ---

	// --- Direct Implementation ---
	findRelevantComments(targetFileAST, path, cursorPos, fset, info, logger) // Populates info
	return nil                                                               // Errors are added to info.AnalysisErrors internally
}

// ============================================================================
// Preamble Construction Helpers (Cycle 8 Refactor)
// ============================================================================

// constructPromptPreamble builds the final preamble string from analyzed info.
// Conceptual: Could be wrapped by memory cache.
func constructPromptPreamble(
	analyzer *GoPackagesAnalyzer, // For potential caching
	info *AstContextInfo,
	qualifier types.Qualifier,
	logger *slog.Logger,
) string {
	// --- Memory Cache Check (Conceptual) ---
	// ... (cache logic omitted for brevity) ...
	// --- End Cache Check ---

	// --- Direct Implementation ---
	return buildPreamble(analyzer, info, qualifier, logger)
}

// buildPreamble constructs the context string sent to the LLM from analysis info.
// Internal helper called by constructPromptPreamble.
func buildPreamble(
	analyzer *GoPackagesAnalyzer, // Pass down for potential future use
	info *AstContextInfo,
	qualifier types.Qualifier,
	logger *slog.Logger,
) string {
	var preamble strings.Builder
	// Use a reasonable internal limit, final truncation happens in FormatPrompt
	const internalPreambleLimit = 8192
	currentLen := 0

	// Helper to add string to preamble if within limit
	addToPreamble := func(s string) bool {
		if currentLen+len(s) < internalPreambleLimit {
			preamble.WriteString(s)
			currentLen += len(s)
			return true
		}
		logger.Debug("Internal preamble limit reached", "limit", internalPreambleLimit)
		return false // Limit reached
	}

	// Helper to add truncation marker if within limit
	addTruncMarker := func(section string) {
		msg := fmt.Sprintf("//   ... (%s truncated)\n", section)
		if currentLen+len(msg) < internalPreambleLimit {
			preamble.WriteString(msg)
			currentLen += len(msg)
		} else {
			logger.Debug("Preamble limit reached even before adding truncation marker", "section", section)
		}
	}

	// --- Build Preamble Sections ---
	// Add file/package context
	if !addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), info.PackageName)) {
		return preamble.String()
	}

	// Add imports section
	if !formatImportsSection(&preamble, info, addToPreamble, addTruncMarker, logger) {
		return preamble.String()
	}

	// Add enclosing function/method section
	if !formatEnclosingFuncSection(&preamble, info, qualifier, addToPreamble) {
		return preamble.String()
	}

	// Add relevant comments section
	if !formatCommentsSection(&preamble, info, addToPreamble, addTruncMarker, logger) {
		return preamble.String()
	}

	// Add specific cursor context (call, selector, etc.) section
	if !formatCursorContextSection(&preamble, info, qualifier, addToPreamble, logger) {
		return preamble.String()
	}

	// Add variables/types in scope section
	formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker, logger)

	return preamble.String()
}

// formatImportsSection formats the import list, respecting limits.
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) bool {
	if len(info.Imports) == 0 {
		return true
	} // No imports, section skipped
	if !add("// Imports:\n") {
		return false
	}

	count := 0
	maxImports := 20 // Limit number of imports shown
	for _, imp := range info.Imports {
		if imp == nil || imp.Path == nil {
			continue
		} // Skip nil imports
		if count >= maxImports {
			addTrunc("imports") // Add truncation marker if limit exceeded
			logger.Debug("Truncated imports list in preamble", "max_imports", maxImports)
			return true // Stop processing imports, but section was added
		}
		path := imp.Path.Value // Import path (e.g., "fmt")
		name := ""
		if imp.Name != nil {
			name = imp.Name.Name + " "
		} // Named import (e.g., f "fmt")
		line := fmt.Sprintf("//   import %s%s\n", name, path)
		if !add(line) {
			return false
		} // Limit reached while adding an import
		count++
	}
	return true // Finished adding imports within limits
}

// formatEnclosingFuncSection formats the enclosing function/method info.
func formatEnclosingFuncSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool) bool {
	funcOrMethod := "Function"
	receiverStr := ""
	if info.ReceiverType != "" {
		funcOrMethod = "Method"
		receiverStr = fmt.Sprintf("(%s) ", info.ReceiverType)
	}

	var funcHeader string
	if info.EnclosingFunc != nil { // Prefer type information
		name := info.EnclosingFunc.Name()
		sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		// Clean up signature string for methods
		if info.ReceiverType != "" && strings.HasPrefix(sigStr, "func(") {
			sigStr = strings.TrimPrefix(sigStr, "func") // Keep func keyword for clarity? Maybe remove only first 'func'? Let's keep func for now.
		}
		funcHeader = fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr)
	} else if info.EnclosingFuncNode != nil { // Fallback to AST node
		name := "[anonymous]"
		if info.EnclosingFuncNode.Name != nil {
			name = info.EnclosingFuncNode.Name.Name
		}
		// Show only basic signature from AST
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s(...)\n", funcOrMethod, receiverStr, name)
	} else {
		return true // No enclosing function found
	}
	return add(funcHeader) // Add the formatted header
}

// formatCommentsSection formats relevant comments, respecting limits.
func formatCommentsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) bool {
	if len(info.CommentsNearCursor) == 0 {
		return true
	}
	if !add("// Relevant Comments:\n") {
		return false
	}

	count := 0
	maxComments := 5 // Limit number of comments shown
	for _, c := range info.CommentsNearCursor {
		if count >= maxComments {
			addTrunc("comments") // Add truncation marker
			logger.Debug("Truncated comments list in preamble", "max_comments", maxComments)
			return true // Stop processing comments
		}
		// Clean up comment markers
		cleanComment := strings.TrimSpace(strings.TrimPrefix(c, "//"))
		cleanComment = strings.TrimSpace(strings.TrimPrefix(cleanComment, "/*"))
		cleanComment = strings.TrimSpace(strings.TrimSuffix(cleanComment, "*/"))

		if len(cleanComment) > 0 {
			line := fmt.Sprintf("//   %s\n", cleanComment) // Add cleaned comment line
			if !add(line) {
				return false
			} // Limit reached while adding a comment
			count++
		}
	}
	return true // Finished adding comments within limits
}

// formatCursorContextSection formats specific context like calls, selectors, etc.
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, logger *slog.Logger) bool {
	// --- Function Call Context ---
	if info.CallExpr != nil {
		// ... (logic as before, using add() and qualifier) ...
		funcName := "[unknown function]"
		switch fun := info.CallExpr.Fun.(type) {
		case *ast.Ident:
			funcName = fun.Name
		case *ast.SelectorExpr:
			if fun.Sel != nil {
				funcName = fun.Sel.Name
			}
		}
		if !add(fmt.Sprintf("// Inside function call: %s (Arg %d)\n", funcName, info.CallArgIndex+1)) {
			return false
		}
		if sig := info.CallExprFuncType; sig != nil {
			if !add(fmt.Sprintf("// Function Signature: %s\n", types.TypeString(sig, qualifier))) {
				return false
			}
			params := sig.Params()
			if params != nil && params.Len() > 0 {
				if !add("//   Parameters:\n") {
					return false
				}
				for i := 0; i < params.Len(); i++ {
					p := params.At(i)
					if p == nil {
						continue
					}
					highlight := ""
					isVariadic := sig.Variadic() && i == params.Len()-1
					if i == info.CallArgIndex || (isVariadic && info.CallArgIndex >= i) {
						highlight = " // <-- cursor here"
						if isVariadic {
							highlight += " (variadic)"
						}
					}
					if !add(fmt.Sprintf("//     - %s%s\n", types.ObjectString(p, qualifier), highlight)) {
						return false
					}
				}
			} else {
				if !add("//   Parameters: (none)\n") {
					return false
				}
			}
			results := sig.Results()
			if results != nil && results.Len() > 0 {
				if !add("//   Returns:\n") {
					return false
				}
				for i := 0; i < results.Len(); i++ {
					r := results.At(i)
					if r == nil {
						continue
					}
					if !add(fmt.Sprintf("//     - %s\n", types.ObjectString(r, qualifier))) {
						return false
					}
				}
			} else {
				if !add("//   Returns: (none)\n") {
					return false
				}
			}
		} else {
			if !add("// Function Signature: (unknown - type analysis failed for call expression)\n") {
				return false
			}
		}
		return true // Handled call expression context
	}

	// --- Selector Expression Context ---
	if info.SelectorExpr != nil {
		// ... (logic as before, using add() and qualifier, calling listTypeMembers) ...
		selName := ""
		if info.SelectorExpr.Sel != nil {
			selName = info.SelectorExpr.Sel.Name
		}
		typeName := "(unknown - type analysis failed for base expression)"
		if info.SelectorExprType != nil {
			typeName = types.TypeString(info.SelectorExprType, qualifier)
		}
		fieldOrMethod, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
		isKnownMember := fieldOrMethod != nil
		unknownMemberMsg := ""
		if info.SelectorExprType != nil && !isKnownMember && selName != "" {
			unknownMemberMsg = fmt.Sprintf(" (unknown member '%s')", selName)
		}
		if !add(fmt.Sprintf("// Selector context: expr type = %s%s\n", typeName, unknownMemberMsg)) {
			return false
		}
		if info.SelectorExprType != nil {
			// List members if base type known AND (member is known OR selection is incomplete)
			if isKnownMember || selName == "" {
				members := listTypeMembers(info.SelectorExprType, info.SelectorExpr.X, qualifier, logger) // Pass logger
				var fields, methods []MemberInfo
				if members != nil {
					for _, m := range members {
						if m.Kind == FieldMember {
							fields = append(fields, m)
						}
						if m.Kind == MethodMember {
							methods = append(methods, m)
						}
					}
				}
				if len(fields) > 0 {
					if !add("//   Available Fields:\n") {
						return false
					}
					sort.Slice(fields, func(i, j int) bool { return fields[i].Name < fields[j].Name })
					for _, field := range fields {
						if !add(fmt.Sprintf("//     - %s %s\n", field.Name, field.TypeString)) {
							return false
						}
					}
				}
				if len(methods) > 0 {
					if !add("//   Available Methods:\n") {
						return false
					}
					sort.Slice(methods, func(i, j int) bool { return methods[i].Name < methods[j].Name })
					for _, method := range methods {
						methodSig := strings.TrimPrefix(method.TypeString, "func")
						if !add(fmt.Sprintf("//     - %s%s\n", method.Name, methodSig)) {
							return false
						}
					}
				}
				if len(fields) == 0 && len(methods) == 0 {
					msg := "//   (No exported fields or methods found)\n"
					if members == nil {
						msg = "//   (Could not determine members)\n"
					}
					if !add(msg) {
						return false
					}
				}
			} else { // Selected member is explicitly unknown
				if !add("//   (Cannot list members: selected member is unknown)\n") {
					return false
				}
			}
		} else {
			if !add("//   (Cannot list members: type analysis failed for base expression)\n") {
				return false
			}
		}
		return true // Handled selector expression context
	}

	// --- Composite Literal Context ---
	if info.CompositeLit != nil {
		// ... (logic as before, using add() and qualifier, calling listStructFields) ...
		typeName := "(unknown - type analysis failed for literal)"
		if info.CompositeLitType != nil {
			typeName = types.TypeString(info.CompositeLitType, qualifier)
		}
		if !add(fmt.Sprintf("// Inside composite literal of type: %s\n", typeName)) {
			return false
		}
		if info.CompositeLitType != nil {
			var st *types.Struct
			currentType := info.CompositeLitType.Underlying()
			if ptr, ok := currentType.(*types.Pointer); ok {
				if ptr.Elem() != nil {
					currentType = ptr.Elem().Underlying()
				} else {
					currentType = nil
				}
			}
			st, _ = currentType.(*types.Struct)
			if st != nil {
				presentFields := make(map[string]bool)
				for _, elt := range info.CompositeLit.Elts {
					if kv, ok := elt.(*ast.KeyValueExpr); ok {
						if kid, ok := kv.Key.(*ast.Ident); ok {
							presentFields[kid.Name] = true
						}
					}
				}
				var missingFields []string
				for i := 0; i < st.NumFields(); i++ {
					field := st.Field(i)
					if field != nil && field.Exported() && !presentFields[field.Name()] {
						missingFields = append(missingFields, types.ObjectString(field, qualifier))
					}
				}
				if len(missingFields) > 0 {
					if !add("//   Missing Exported Fields (candidates for completion):\n") {
						return false
					}
					sort.Strings(missingFields)
					for _, fieldStr := range missingFields {
						if !add(fmt.Sprintf("//     - %s\n", fieldStr)) {
							return false
						}
					}
				} else {
					if !add("//   (All exported fields may be present or none missing)\n") {
						return false
					}
				}
			} else {
				if !add("//   (Underlying type is not a struct)\n") {
					return false
				}
			}
		} else {
			if !add("//   (Cannot determine missing fields: type analysis failed)\n") {
				return false
			}
		}
		return true // Handled composite literal context
	}

	// --- Identifier Context ---
	if info.IdentifierAtCursor != nil {
		identName := info.IdentifierAtCursor.Name
		identTypeStr := "(Type unknown)"
		if info.IdentifierType != nil {
			identTypeStr = fmt.Sprintf("(Type: %s)", types.TypeString(info.IdentifierType, qualifier))
		}
		if !add(fmt.Sprintf("// Identifier at cursor: %s %s\n", identName, identTypeStr)) {
			return false
		}
		return true // Handled identifier context
	}

	return true // No specific cursor context found or limit reached earlier
}

// formatScopeSection formats variables/constants/types in scope, respecting limits.
func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string), logger *slog.Logger) bool {
	if len(info.VariablesInScope) == 0 {
		return true
	}
	if !add("// Variables/Constants/Types in Scope:\n") {
		return false
	}

	var items []string
	for name := range info.VariablesInScope {
		obj := info.VariablesInScope[name]
		items = append(items, fmt.Sprintf("//   %s\n", types.ObjectString(obj, qualifier))) // Use types.ObjectString
	}
	sort.Strings(items) // Sort for consistent order

	count := 0
	maxScopeItems := 30 // Limit number of scope items shown
	for _, item := range items {
		if count >= maxScopeItems {
			addTrunc("scope") // Add truncation marker
			logger.Debug("Truncated scope list in preamble", "max_items", maxScopeItems)
			return true // Stop processing scope items
		}
		if !add(item) {
			return false
		} // Limit reached while adding a scope item
		count++
	}
	return true // Finished adding scope items within limits
}

// ============================================================================
// Core Analysis Logic Helpers (Cycle 8 Refactor)
// ============================================================================

// addAnalysisError adds a non-fatal error to the info struct, avoiding duplicates.
func addAnalysisError(info *AstContextInfo, err error, logger *slog.Logger) {
	if err == nil || info == nil {
		return
	} // No error or nowhere to add it
	errMsg := err.Error()
	// Check for duplicates before adding
	for _, existing := range info.AnalysisErrors {
		if existing.Error() == errMsg {
			return
		} // Avoid duplicate error messages
	}
	logger.Warn("Analysis warning", "error", err) // Log the warning
	info.AnalysisErrors = append(info.AnalysisErrors, err)
}

// logAnalysisErrors logs joined non-fatal analysis errors if any occurred.
func logAnalysisErrors(errs []error, logger *slog.Logger) {
	if len(errs) > 0 {
		combinedErr := errors.Join(errs...) // Use errors.Join
		logger.Warn("Context analysis completed with non-fatal errors", "count", len(errs), "errors", combinedErr)
	}
}

// findEnclosingPath finds the AST node path from the root to the node enclosing the cursor.
func findEnclosingPath(targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo, logger *slog.Logger) []ast.Node {
	if targetFileAST == nil {
		addAnalysisError(info, errors.New("cannot find enclosing path: targetFileAST is nil"), logger)
		return nil
	}
	if !cursorPos.IsValid() {
		addAnalysisError(info, errors.New("cannot find enclosing path: invalid cursor position"), logger)
		return nil
	}
	// astutil.PathEnclosingInterval finds the tightest enclosing node sequence
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		logger.Debug("No AST path found enclosing cursor position", "pos", cursorPos)
	}
	return path
}

// gatherScopeContext walks the enclosing path to find relevant scope information.
func gatherScopeContext(path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	// Add package scope first (widest scope)
	addPackageScope(targetPkg, info, logger)

	// Walk up the AST path, adding variables from enclosing blocks/functions
	if path != nil {
		for i := len(path) - 1; i >= 0; i-- { // Iterate from outermost to innermost relevant scope
			node := path[i]
			switch n := node.(type) {
			case *ast.File: // File scope already handled by addPackageScope
				continue
			case *ast.FuncDecl:
				// Add function parameters and named results to scope
				if info.EnclosingFuncNode == nil { // Capture the first FuncDecl encountered walking up
					info.EnclosingFuncNode = n
					// Try to get receiver type string from AST if fset is available
					if fset != nil && n.Recv != nil && len(n.Recv.List) > 0 && n.Recv.List[0].Type != nil {
						var buf bytes.Buffer
						if err := format.Node(&buf, fset, n.Recv.List[0].Type); err == nil {
							info.ReceiverType = buf.String()
						} else {
							logger.Warn("could not format receiver type", "error", err)
							info.ReceiverType = "[error formatting receiver]"
							addAnalysisError(info, fmt.Errorf("receiver format error: %w", err), logger)
						}
					}
				}
				// Try to get type info for the function
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
					if obj, ok := targetPkg.TypesInfo.Defs[n.Name]; ok && obj != nil {
						if fn, ok := obj.(*types.Func); ok {
							if info.EnclosingFunc == nil { // Capture the first Func type object
								info.EnclosingFunc = fn
							}
							// Add signature parameters/results to the current scope map
							if sig, ok := fn.Type().(*types.Signature); ok {
								addSignatureToScope(sig, info.VariablesInScope)
							}
						}
					} else if info.EnclosingFunc == nil { // Only log error if we haven't found type info yet
						addAnalysisError(info, fmt.Errorf("definition for func '%s' not found in TypesInfo", n.Name.Name), logger)
					}
				} else if info.EnclosingFunc == nil { // Log if type info is missing and we need it
					reason := getMissingTypeInfoReason(targetPkg)
					funcName := "[anonymous]"
					if n.Name != nil {
						funcName = n.Name.Name
					}
					addAnalysisError(info, fmt.Errorf("type info for enclosing func '%s' unavailable: %s", funcName, reason), logger)
				}

			case *ast.BlockStmt:
				// Add variables declared in this block
				if info.EnclosingBlock == nil {
					info.EnclosingBlock = n
				} // Capture the innermost block
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						// Add variables declared *before* the cursor within this block
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					} else { // Scope info missing for this node
						posStr := getPosString(fset, n.Pos())
						addAnalysisError(info, fmt.Errorf("scope info missing for block at %s", posStr), logger)
					}
				} else if info.EnclosingBlock == n { // Log if type info is missing for the innermost block
					reason := getMissingTypeInfoReason(targetPkg)
					posStr := getPosString(fset, n.Pos())
					addAnalysisError(info, fmt.Errorf("cannot get scope variables for block at %s: %s", posStr, reason), logger)
				}
				// Add other scope-introducing nodes if necessary (e.g., *ast.IfStmt, *ast.ForStmt)
			}
		}
	} else {
		logger.Debug("AST path is nil, cannot gather block/function scopes.")
	}
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

// addPackageScope adds package-level identifiers to the scope map.
func addPackageScope(targetPkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope) // cursorPos doesn't apply
		} else {
			addAnalysisError(info, fmt.Errorf("package scope missing for pkg %s", targetPkg.PkgPath), logger)
		}
	} else {
		reason := "targetPkg is nil"
		if targetPkg != nil {
			reason = fmt.Sprintf("package.Types field is nil for pkg %s", targetPkg.PkgPath)
		}
		addAnalysisError(info, fmt.Errorf("cannot add package scope: %s", reason), logger)
	}
}

// addScopeVariables adds identifiers from a types.Scope if declared before cursorPos.
func addScopeVariables(typeScope *types.Scope, cursorPos token.Pos, scopeMap map[string]types.Object) {
	if typeScope == nil {
		return
	}
	for _, name := range typeScope.Names() {
		obj := typeScope.Lookup(name)
		if obj == nil {
			continue
		}
		// Include if cursor invalid OR object pos invalid OR object declared before cursor
		include := !cursorPos.IsValid() || !obj.Pos().IsValid() || obj.Pos() < cursorPos
		if include {
			if _, exists := scopeMap[name]; !exists { // Inner scopes override outer
				switch obj.(type) {
				case *types.Var, *types.Const, *types.TypeName, *types.Func, *types.PkgName, *types.Builtin, *types.Nil:
					scopeMap[name] = obj
				}
			}
		}
	}
}

// addSignatureToScope adds named parameters and results to the scope map.
func addSignatureToScope(sig *types.Signature, scopeMap map[string]types.Object) {
	if sig == nil {
		return
	}
	addTupleToScope(sig.Params(), scopeMap)
	addTupleToScope(sig.Results(), scopeMap)
}

// addTupleToScope adds named variables from a types.Tuple to the scope map.
func addTupleToScope(tuple *types.Tuple, scopeMap map[string]types.Object) {
	if tuple == nil {
		return
	}
	for j := 0; j < tuple.Len(); j++ {
		v := tuple.At(j)
		if v != nil && v.Name() != "" { // Add only named variables
			if _, exists := scopeMap[v.Name()]; !exists {
				scopeMap[v.Name()] = v
			}
		}
	}
}

// findRelevantComments uses ast.CommentMap to find comments near the cursor.
func findRelevantComments(targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if targetFileAST == nil || fset == nil {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"), logger)
		return
	}
	// Create a comment map for efficient lookup
	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset, logger)
}

// findCommentsWithMap implements the logic to find preceding or enclosing doc comments.
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, logger *slog.Logger) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		logger.Debug("Skipping comment finding due to nil cmap, invalid cursor, or nil fset")
		return comments
	}

	cursorLine := fset.Position(cursorPos).Line
	var precedingComments []string

	// Strategy 1: Find comments immediately preceding the cursor line.
	// Iterate through the comment map (less efficient but simple).
	// TODO: Optimization: Could potentially use the AST path to find the node immediately
	//       before the cursor and check its preceding comments directly.
	foundPrecedingOnLine := false
	minCommentPos := token.Pos(-1) // Track the closest preceding comment start

	for node, groups := range cmap {
		if node == nil {
			continue
		}
		for _, cg := range groups {
			if cg == nil || len(cg.List) == 0 {
				continue
			}

			// Check if the comment group ENDS on the line immediately before the cursor
			commentEndLine := fset.Position(cg.End()).Line
			if commentEndLine == cursorLine-1 {
				// Check if this comment is closer than previously found ones on this line
				if !foundPrecedingOnLine || cg.Pos() > minCommentPos {
					precedingComments = nil // Reset if we find a closer group on the same line
					for _, c := range cg.List {
						if c != nil {
							precedingComments = append(precedingComments, c.Text)
						}
					}
					minCommentPos = cg.Pos()
					foundPrecedingOnLine = true
				}
			}
		}
	}

	// If comments were found immediately preceding the line, use them.
	if foundPrecedingOnLine {
		logger.Debug("Found preceding comments on line before cursor", "count", len(precedingComments))
		comments = append(comments, precedingComments...)
	} else {
		// Strategy 2: If no immediately preceding comments, find the doc comment
		//             of the first enclosing declaration (func, type, var, const).
		logger.Debug("No comments found on preceding line, looking for enclosing doc comments.")
		if path != nil {
			for i := 0; i < len(path); i++ { // Iterate from innermost to outermost
				node := path[i]
				var docComment *ast.CommentGroup
				// Check node types that typically have doc comments
				switch n := node.(type) {
				case *ast.FuncDecl:
					docComment = n.Doc
				case *ast.GenDecl:
					docComment = n.Doc // Covers var, const, type blocks
				case *ast.TypeSpec:
					docComment = n.Doc // Individual type specs
				case *ast.Field:
					docComment = n.Doc // Struct fields
				case *ast.ValueSpec:
					docComment = n.Doc // Individual var/const specs
				}
				if docComment != nil {
					logger.Debug("Found doc comment on enclosing node", "node_type", fmt.Sprintf("%T", node))
					for _, c := range docComment.List {
						if c != nil {
							comments = append(comments, c.Text)
						}
					}
					goto cleanup // Use the first doc comment found walking up
				}
			}
		}
	}

cleanup:
	// Remove duplicate comments (e.g., if preceding comment was also doc comment)
	if len(comments) > 1 {
		seen := make(map[string]struct{})
		uniqueComments := make([]string, 0, len(comments))
		for _, c := range comments {
			if _, ok := seen[c]; !ok {
				seen[c] = struct{}{}
				uniqueComments = append(uniqueComments, c)
			}
		}
		comments = uniqueComments
	}
	logger.Debug("Final relevant comments found", "count", len(comments))
	return comments
}

// findContextNodes identifies specific AST nodes and populates info struct.
// Updated for Hover refinement (stores IdentifierDefNode).
func findContextNodes(
	path []ast.Node,
	cursorPos token.Pos,
	pkg *packages.Package,
	fset *token.FileSet,
	analyzer *GoPackagesAnalyzer, // For potential caching
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) {
	if len(path) == 0 || fset == nil {
		if fset == nil {
			addAnalysisError(info, errors.New("Fset is nil in findContextNodes"), logger)
		}
		return // Cannot determine context without path or fileset
	}

	posStr := func(p token.Pos) string { return getPosString(fset, p) }
	hasTypeInfo := pkg != nil && pkg.TypesInfo != nil
	var typesMap map[ast.Expr]types.TypeAndValue
	var defsMap map[*ast.Ident]types.Object
	var usesMap map[*ast.Ident]types.Object
	if hasTypeInfo {
		typesMap = pkg.TypesInfo.Types
		defsMap = pkg.TypesInfo.Defs
		usesMap = pkg.TypesInfo.Uses
		if typesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Types' is nil"), logger)
		}
		if defsMap == nil {
			addAnalysisError(info, errors.New("type info map 'Defs' is nil"), logger)
		}
		if usesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Uses' is nil"), logger)
		}
	} else {
		reason := getMissingTypeInfoReason(pkg)
		addAnalysisError(info, fmt.Errorf("cannot perform type analysis: %s", reason), logger)
	}

	// --- Check for Composite Literal ---
	innermostNode := path[0]
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
		logger.Debug("Cursor inside Composite Literal", "pos", posStr(compLit.Pos()))
		info.CompositeLit = compLit
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[compLit]; ok {
				info.CompositeLitType = tv.Type
				if info.CompositeLitType == nil {
					addAnalysisError(info, fmt.Errorf("composite literal type resolved to nil at %s", posStr(compLit.Pos())), logger)
				}
			} else {
				addAnalysisError(info, fmt.Errorf("missing type info for composite literal at %s", posStr(compLit.Pos())), logger)
			}
		}
		return // Found context, exit
	}

	// --- Check for Function Call ---
	var callExpr *ast.CallExpr
	if ce, ok := path[0].(*ast.CallExpr); ok {
		callExpr = ce
	} else if len(path) > 1 {
		if ce, ok := path[1].(*ast.CallExpr); ok {
			callExpr = ce
		}
	}
	if callExpr != nil && cursorPos > callExpr.Lparen && cursorPos <= callExpr.Rparen {
		logger.Debug("Cursor inside Call Expression arguments", "pos", posStr(callExpr.Pos()))
		info.CallExpr = callExpr
		info.CallArgIndex = calculateArgIndex(callExpr.Args, cursorPos)
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[callExpr.Fun]; ok && tv.Type != nil {
				if sig, ok := tv.Type.Underlying().(*types.Signature); ok {
					info.CallExprFuncType = sig
					info.ExpectedArgType = determineExpectedArgType(sig, info.CallArgIndex)
				} else {
					addAnalysisError(info, fmt.Errorf("type of call func (%T) at %s is %T, not signature", callExpr.Fun, posStr(callExpr.Fun.Pos()), tv.Type), logger)
				}
			} else {
				reason := "missing type info entry"
				if ok && tv.Type == nil {
					reason = "type info resolved to nil"
				}
				addAnalysisError(info, fmt.Errorf("%s for call func (%T) at %s", reason, callExpr.Fun, posStr(callExpr.Fun.Pos())), logger)
			}
		}
		return // Found context, exit
	}

	// --- Check for Selector Expression ---
	for i := 0; i < len(path) && i < 2; i++ {
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok && cursorPos > selExpr.X.End() {
			logger.Debug("Cursor inside Selector Expression", "pos", posStr(selExpr.Pos()))
			info.SelectorExpr = selExpr
			if hasTypeInfo && typesMap != nil {
				if tv, ok := typesMap[selExpr.X]; ok {
					info.SelectorExprType = tv.Type
					if tv.Type == nil {
						addAnalysisError(info, fmt.Errorf("type info resolved to nil for selector base expr (%T) starting at %s", selExpr.X, posStr(selExpr.X.Pos())), logger)
					}
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info entry for selector base expr (%T) starting at %s", selExpr.X, posStr(selExpr.X.Pos())), logger)
				}
			}
			// Check if selected member is known
			if info.SelectorExprType != nil && selExpr.Sel != nil {
				selName := selExpr.Sel.Name
				obj, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
				if obj == nil {
					addAnalysisError(info, fmt.Errorf("selecting unknown member '%s' from type '%s'", selName, info.SelectorExprType.String()), logger)
				}
			}
			return // Found context, exit
		}
	}

	// --- Check for Identifier ---
	var ident *ast.Ident
	if id, ok := path[0].(*ast.Ident); ok && cursorPos >= id.Pos() && cursorPos <= id.End() { // Cursor within or at end
		// Ensure it's not the 'Sel' part of a selector we already handled
		isSelectorSel := false
		if len(path) > 1 {
			if sel, ok := path[1].(*ast.SelectorExpr); ok && sel.Sel == id {
				isSelectorSel = true
			}
		}
		if !isSelectorSel {
			ident = id
		}
	}
	// Consider parent if innermost wasn't an ident (less common case)
	// else if len(path) > 1 { ... logic from previous version ... }

	if ident != nil {
		logger.Debug("Cursor at Identifier", "name", ident.Name, "pos", posStr(ident.Pos()))
		info.IdentifierAtCursor = ident
		if hasTypeInfo {
			var obj types.Object
			if usesMap != nil {
				obj = usesMap[ident]
			}
			if obj == nil && defsMap != nil {
				obj = defsMap[ident]
			}

			if obj != nil {
				info.IdentifierObject = obj
				info.IdentifierType = obj.Type()
				if info.IdentifierType == nil {
					addAnalysisError(info, fmt.Errorf("object '%s' at %s found but type is nil", obj.Name(), posStr(obj.Pos())), logger)
				}

				// --- ** Cycle 2: Hover Refinement: Find Defining Node ** ---
				defPos := obj.Pos()
				if defPos.IsValid() {
					// Find the AST file containing the definition
					defFile := fset.File(defPos)
					if defFile == nil {
						addAnalysisError(info, fmt.Errorf("could not find token.File for definition of '%s' at pos %s", obj.Name(), posStr(defPos)), logger)
					} else {
						var defAST *ast.File
						// Check if definition is in the currently analyzed file
						if defFile.Name() == info.FilePath {
							defAST = info.TargetAstFile
						} else {
							// Definition is in another file in the package. Find its AST.
							if pkg != nil {
								for _, syntaxFile := range pkg.Syntax {
									if fset.File(syntaxFile.Pos()) == defFile {
										defAST = syntaxFile
										break
									}
								}
							}
						}

						if defAST != nil {
							// Find the specific AST node at the definition position
							defPath, _ := astutil.PathEnclosingInterval(defAST, defPos, defPos)
							if len(defPath) > 0 {
								// Find the most specific node that corresponds to the definition
								// (e.g., FuncDecl, ValueSpec, TypeSpec, Field)
								for _, node := range defPath {
									// Check if the node's position matches the object's position
									// and if it's a suitable declaration/spec node.
									isDeclNode := false
									switch n := node.(type) {
									case *ast.FuncDecl:
										if n.Name != nil && n.Name.Pos() == defPos {
											isDeclNode = true
										}
									case *ast.ValueSpec: // var, const
										for _, name := range n.Names {
											if name != nil && name.Pos() == defPos {
												isDeclNode = true
												break
											}
										}
									case *ast.TypeSpec: // type
										if n.Name != nil && n.Name.Pos() == defPos {
											isDeclNode = true
										}
									case *ast.Field: // struct field, interface method, func param/result
										for _, name := range n.Names {
											if name != nil && name.Pos() == defPos {
												isDeclNode = true
												break
											}
										}
									}
									if isDeclNode {
										info.IdentifierDefNode = node
										logger.Debug("Found defining AST node for hover", "object", obj.Name(), "node_type", fmt.Sprintf("%T", node), "pos", posStr(node.Pos()))
										break // Found the most specific declaration node
									}
								}
								if info.IdentifierDefNode == nil {
									// Could not find a specific decl node, maybe use innermost node at defPos?
									info.IdentifierDefNode = defPath[0]
									logger.Warn("Could not pinpoint specific defining declaration node, using innermost node at definition position", "object", obj.Name(), "innermost_type", fmt.Sprintf("%T", defPath[0]))
									addAnalysisError(info, fmt.Errorf("could not find specific defining node for '%s' at pos %s", obj.Name(), posStr(defPos)), logger)
								}
							} else {
								addAnalysisError(info, fmt.Errorf("could not find AST path for definition of '%s' at pos %s", obj.Name(), posStr(defPos)), logger)
							}
						} else {
							addAnalysisError(info, fmt.Errorf("could not find AST for definition file '%s' of object '%s'", defFile.Name(), obj.Name()), logger)
						}
					}
				} else {
					logger.Debug("Object has invalid definition position, cannot find defining node.", "object", obj.Name())
				}
				// --- End Hover Refinement ---

			} else { // Object not found in defs or uses
				if typesMap != nil {
					if tv, ok := typesMap[ident]; ok && tv.Type != nil {
						info.IdentifierType = tv.Type
					} else {
						addAnalysisError(info, fmt.Errorf("missing object and type info for identifier '%s' at %s", ident.Name, posStr(ident.Pos())), logger)
					}
				} else {
					addAnalysisError(info, fmt.Errorf("object not found for identifier '%s' at %s (and Types map is nil)", ident.Name, posStr(ident.Pos())), logger)
				}
			}
		} else {
			addAnalysisError(info, errors.New("missing type info for identifier analysis"), logger)
		}
		// No return here, allow other contexts to be checked if needed
	}
	logger.Debug("Finished context node identification.")
}

// calculateArgIndex determines the 0-based index of the argument the cursor is in.
func calculateArgIndex(args []ast.Expr, cursorPos token.Pos) int {
	if len(args) == 0 {
		return 0
	} // No args, cursor is effectively at index 0
	for i, arg := range args {
		if arg == nil {
			continue
		}
		argStart := arg.Pos()
		argEnd := arg.End()
		slotStart := argStart
		if i > 0 && args[i-1] != nil {
			slotStart = args[i-1].End() + 1
		} // Slot starts after previous arg/comma

		// Cursor is between start of this arg's slot and end of this arg
		if cursorPos >= slotStart && cursorPos <= argEnd {
			return i
		}
		// Cursor is after this arg's end
		if cursorPos > argEnd {
			// If this is the last arg, cursor is position for next arg
			if i == len(args)-1 {
				return i + 1
			}
			// If cursor is before the start of the *next* arg, it's in the slot for the next arg
			if args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i + 1
			}
		}
	}
	// Cursor is before the first argument
	if len(args) > 0 && args[0] != nil && cursorPos < args[0].Pos() {
		return 0
	}
	// Default fallback (e.g., cursor after last arg's comma)
	return len(args)
}

// determineExpectedArgType finds the expected type for a given argument index in a signature.
func determineExpectedArgType(sig *types.Signature, argIndex int) types.Type {
	if sig == nil || argIndex < 0 {
		return nil
	}
	params := sig.Params()
	if params == nil {
		return nil
	}
	numParams := params.Len()
	if numParams == 0 {
		return nil
	}

	if sig.Variadic() {
		if argIndex >= numParams-1 { // At or after variadic param
			lastParam := params.At(numParams - 1)
			if lastParam == nil {
				return nil
			}
			if slice, ok := lastParam.Type().(*types.Slice); ok {
				return slice.Elem()
			} // Expect element type
			return nil // Should be slice
		}
		// Before variadic param
		param := params.At(argIndex)
		if param == nil {
			return nil
		}
		return param.Type()
	} else { // Not variadic
		if argIndex < numParams {
			param := params.At(argIndex)
			if param == nil {
				return nil
			}
			return param.Type()
		}
	}
	return nil // Index out of bounds
}

// listTypeMembers attempts to list exported fields and methods for a given type.
func listTypeMembers(typ types.Type, expr ast.Expr, qualifier types.Qualifier, logger *slog.Logger) []MemberInfo {
	if typ == nil {
		logger.Debug("Cannot list members: input type is nil")
		return nil
	}
	logger = logger.With("type", typ.String())

	var members []MemberInfo
	seenMembers := make(map[string]MemberKind) // Track members to avoid duplicates

	// --- Get methods from the method set ---
	msets := []*types.MethodSet{types.NewMethodSet(typ)}
	if _, isInterface := typ.Underlying().(*types.Interface); !isInterface {
		if ptrType := types.NewPointer(typ); ptrType != nil {
			msets = append(msets, types.NewMethodSet(ptrType))
		}
	}

	for _, mset := range msets {
		for i := 0; i < mset.Len(); i++ {
			sel := mset.At(i)
			if sel == nil {
				continue
			}
			methodObj := sel.Obj()
			if method, ok := methodObj.(*types.Func); ok && method != nil && method.Exported() {
				methodName := method.Name()
				if _, exists := seenMembers[methodName]; !exists {
					members = append(members, MemberInfo{Name: methodName, Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
					seenMembers[methodName] = MethodMember
					logger.Debug("Added method", "name", methodName)
				}
			}
		}
	}

	// --- Get fields from struct type (if applicable) ---
	currentType := typ
	if ptr, ok := typ.(*types.Pointer); ok {
		if ptr.Elem() == nil {
			logger.Debug("Cannot list fields: pointer element type is nil")
			return members
		}
		currentType = ptr.Elem()
	}
	underlying := currentType.Underlying()

	if st, ok := underlying.(*types.Struct); ok {
		logger.Debug("Type is a struct, listing fields", "num_fields", st.NumFields())
		for i := 0; i < st.NumFields(); i++ {
			field := st.Field(i)
			if field != nil && field.Exported() {
				fieldName := field.Name()
				if _, exists := seenMembers[fieldName]; !exists { // Add only if not shadowed by method
					members = append(members, MemberInfo{Name: fieldName, Kind: FieldMember, TypeString: types.TypeString(field.Type(), qualifier)})
					seenMembers[fieldName] = FieldMember
					logger.Debug("Added field", "name", fieldName)
				}
			}
			// TODO: Handle embedded fields recursively if needed?
		}
	} else if iface, ok := underlying.(*types.Interface); ok {
		logger.Debug("Type is an interface, listing explicit methods", "num_methods", iface.NumExplicitMethods())
		for i := 0; i < iface.NumExplicitMethods(); i++ {
			method := iface.ExplicitMethod(i)
			if method != nil && method.Exported() {
				methodName := method.Name()
				if _, exists := seenMembers[methodName]; !exists {
					members = append(members, MemberInfo{Name: methodName, Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
					seenMembers[methodName] = MethodMember
					logger.Debug("Added interface method", "name", methodName)
				}
			}
		}
	} else {
		logger.Debug("Type is not a struct or interface, no fields/explicit methods to list directly.", "type", fmt.Sprintf("%T", underlying))
	}

	logger.Debug("Finished listing members", "count", len(members))
	return members
}

// listStructFields lists exported fields of a struct type. (Potentially redundant with listTypeMembers)
func listStructFields(st *types.Struct, qualifier types.Qualifier) []MemberInfo {
	var fields []MemberInfo
	if st == nil {
		return nil
	}
	for i := 0; i < st.NumFields(); i++ {
		field := st.Field(i)
		if field != nil && field.Exported() {
			fields = append(fields, MemberInfo{Name: field.Name(), Kind: FieldMember, TypeString: types.TypeString(field.Type(), qualifier)})
		}
	}
	return fields
}

// ============================================================================
// Hover Formatting Helper (Cycle 2 Implementation)
// ============================================================================

// formatObjectForHover creates a Markdown string for hover info.
func formatObjectForHover(obj types.Object, info *AstContextInfo, logger *slog.Logger) string {
	if obj == nil {
		logger.Debug("formatObjectForHover called with nil object")
		return ""
	}

	var hoverText strings.Builder

	// --- 1. Format Definition ---
	var qualifier types.Qualifier
	if info.TargetPackage != nil && info.TargetPackage.Types != nil {
		qualifier = types.RelativeTo(info.TargetPackage.Types)
	} else {
		qualifier = func(other *types.Package) string {
			if other != nil {
				return other.Name()
			}
			return ""
		} // Fallback
	}
	definition := types.ObjectString(obj, qualifier)
	if definition != "" {
		hoverText.WriteString("```go\n")
		hoverText.WriteString(definition)
		hoverText.WriteString("\n```") // End code block
	} else {
		logger.Warn("Could not format object definition string", "object", obj.Name())
	}

	// --- 2. Find and Format Documentation ---
	docComment := ""
	var commentGroup *ast.CommentGroup

	// Attempt to get comment from the definition node found earlier
	if info.IdentifierDefNode != nil {
		switch n := info.IdentifierDefNode.(type) {
		case *ast.FuncDecl:
			commentGroup = n.Doc
		case *ast.GenDecl: // Handles var, const, type blocks
			// For GenDecl, find the specific Spec that matches the object's position
			// This is more accurate than just using n.Doc, which applies to the whole block.
			for _, spec := range n.Specs {
				switch s := spec.(type) {
				case *ast.ValueSpec: // var, const
					for _, name := range s.Names {
						if name.Pos() == obj.Pos() {
							commentGroup = s.Doc
							if commentGroup == nil { // Fallback to GenDecl doc if spec doc is nil
								commentGroup = n.Doc
							}
							goto foundDoc // Exit spec loop
						}
					}
				case *ast.TypeSpec: // type
					if s.Name != nil && s.Name.Pos() == obj.Pos() {
						commentGroup = s.Doc
						if commentGroup == nil { // Fallback
							commentGroup = n.Doc
						}
						goto foundDoc // Exit spec loop
					}
				}
			}
			// If no specific spec matched, use the GenDecl doc as a last resort
			if commentGroup == nil {
				commentGroup = n.Doc
			}
		case *ast.TypeSpec: // Handles individual type specs outside GenDecl (less common)
			commentGroup = n.Doc
		case *ast.Field: // Handles struct fields, interface methods, func params/results
			commentGroup = n.Doc
		case *ast.ValueSpec: // Handles individual var/const specs outside GenDecl
			commentGroup = n.Doc
		case *ast.AssignStmt: // Handle short variable declarations (var := value)
			// Check if the object's position matches one of the Lhs identifiers
			for _, lhsExpr := range n.Lhs {
				if ident, ok := lhsExpr.(*ast.Ident); ok && ident.Pos() == obj.Pos() {
					// Short var decls don't have their own .Doc field.
					// We might look for comments *preceding* the AssignStmt in the CommentMap,
					// but that requires passing the CommentMap or re-creating it here.
					// For now, we won't find docs for short var decls this way.
					logger.Debug("Hover object is a short variable declaration; doc comment lookup not implemented for this case.", "object", obj.Name())
					break
				}
			}
		default:
			logger.Debug("Hover documentation lookup: Unhandled definition node type", "type", fmt.Sprintf("%T", n))
		}
	foundDoc: // Label to jump to after finding doc in GenDecl specs
	} else {
		logger.Debug("Defining node (IdentifierDefNode) not found in AstContextInfo", "object", obj.Name())
		// TODO: Fallback? Could try finding comments near obj.Pos() using CommentMap if available?
	}

	// Format the comment group if found
	if commentGroup != nil && len(commentGroup.List) > 0 {
		var doc strings.Builder
		for _, c := range commentGroup.List {
			if c != nil {
				// Basic cleaning: remove comment markers, trim space
				text := strings.TrimSpace(strings.TrimPrefix(c.Text, "//"))
				text = strings.TrimSpace(strings.TrimPrefix(text, "/*"))
				text = strings.TrimSpace(strings.TrimSuffix(text, "*/"))
				if doc.Len() > 0 {
					doc.WriteString("\n") // Add newline between comment lines
				}
				doc.WriteString(text)
			}
		}
		docComment = doc.String()
		logger.Debug("Found and formatted doc comment for object", "object", obj.Name())
	} else if info.IdentifierDefNode != nil {
		logger.Debug("No doc comment found on definition node", "object", obj.Name(), "node_type", fmt.Sprintf("%T", info.IdentifierDefNode))
	}

	// Combine definition and documentation
	if docComment != "" {
		if hoverText.Len() > 0 {
			hoverText.WriteString("\n\n---\n\n") // Separator only if definition exists
		}
		hoverText.WriteString(docComment)
	}

	finalContent := hoverText.String()
	// Avoid returning just an empty code block if definition is empty and no docs found
	if strings.TrimSpace(finalContent) == "```go\n```" {
		return ""
	}
	logger.Debug("Formatted hover content", "object", obj.Name(), "content_length", len(finalContent))
	return finalContent
}

// ============================================================================
// Memory Cache Helpers (Conceptual - Cycle 9)
// ============================================================================

// generateCacheKey creates a key for the memory cache based on context.
// NOTE: This is a placeholder. A robust key needs careful design.
func generateCacheKey(prefix string, info *AstContextInfo) string {
	// Basic key: prefix + filepath + version + cursor position
	// WARNING: Cursor position might be too specific. Caching based on
	// enclosing function/block start position might be better.
	// Hash of relevant AST node might also be needed.
	return fmt.Sprintf("%s:%s:%d:%d", prefix, info.FilePath, info.Version, info.CursorPos)
}

// withMemoryCache wraps a function call with Ristretto caching logic.
// Returns: computed/cached value, cacheHit bool, error during computation.
func withMemoryCache[T any](
	analyzer *GoPackagesAnalyzer,
	cacheKey string,
	cost int64,
	ttl time.Duration,
	computeFn func() (T, error), // Function to compute the value if cache miss
	logger *slog.Logger,
) (T, bool, error) {
	var zero T // Zero value for the return type

	if analyzer == nil || !analyzer.MemoryCacheEnabled() {
		result, err := computeFn()
		return result, false, err // Cache disabled or compute error
	}

	// 1. Check cache
	if cachedResult, found := analyzer.memoryCache.Get(cacheKey); found {
		if typedResult, ok := cachedResult.(T); ok {
			logger.Debug("Ristretto cache hit", "key", cacheKey)
			return typedResult, true, nil // Return cached value
		}
		// Cache contained wrong type - should not happen if keys/types are consistent
		logger.Warn("Ristretto cache type assertion failed", "key", cacheKey, "expected_type", fmt.Sprintf("%T", zero))
		analyzer.memoryCache.Del(cacheKey) // Delete invalid entry
	}
	logger.Debug("Ristretto cache miss", "key", cacheKey)

	// 2. Cache miss: Compute the value
	computedResult, err := computeFn()
	if err != nil {
		return zero, false, err // Error during computation
	}

	// 3. Store computed value in cache (only if no error occurred)
	// Use SetWithTTL for time-based eviction, or Set if relying only on size/cost eviction
	// Defensive: Ensure cost is positive
	if cost <= 0 {
		cost = 1
	}
	setOk := analyzer.memoryCache.SetWithTTL(cacheKey, computedResult, cost, ttl)
	if !setOk {
		// Cache might be full or item cost too high
		logger.Warn("Ristretto cache Set failed, item not cached", "key", cacheKey, "cost", cost)
	} else {
		logger.Debug("Ristretto cache set", "key", cacheKey, "cost", cost, "ttl", ttl)
	}
	// analyzer.memoryCache.Wait() // Optional: Wait for value to be processed by buffer

	return computedResult, false, nil // Return computed value
}
