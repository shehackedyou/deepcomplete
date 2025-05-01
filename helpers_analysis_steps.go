// deepcomplete/helpers_analysis_steps.go
package deepcomplete

import (
	"bytes"
	"context" // Added for potential future context passing
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"log/slog"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

// ============================================================================
// Analysis Step Orchestration & Core Logic Helpers
// ============================================================================

// performAnalysisSteps orchestrates the detailed analysis after loading.
// It populates the AstContextInfo struct with findings and diagnostics.
// Returns an error only if a fatal error occurs preventing further analysis (e.g., invalid cursor).
// Non-fatal errors are added to info.AnalysisErrors.
// ** MODIFIED: Cycle 3 - Improved error handling and flow **
func performAnalysisSteps(
	ctx context.Context, // Added context
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package,
	fset *token.FileSet,
	line, col int, // 1-based line/col from request
	analyzer *GoPackagesAnalyzer, // Needed for memory cache access
	info *AstContextInfo, // Pass info struct to be populated
	logger *slog.Logger,
) error {
	// Defensive check for required inputs
	if fset == nil {
		err := errors.New("performAnalysisSteps requires non-nil fset")
		addAnalysisError(info, err, logger) // Assumes helpers_diagnostics.go exists
		return err                          // Cannot proceed without fileset
	}
	if targetFile == nil {
		// Can still attempt package scope analysis even if target file is missing
		logger.Warn("Target token.File is nil, proceeding with package scope analysis only.")
		addAnalysisError(info, errors.New("target token.File is nil, cannot perform file-specific analysis"), logger)
		gatherScopeContext(ctx, nil, targetPkg, fset, info, logger) // Pass nil path
		return nil                                                  // Not a fatal error for the whole analysis process
	}

	// Calculate cursor position first (0-based token.Pos)
	cursorPos, posErr := calculateCursorPos(targetFile, line, col) // Assumes deepcomplete_utils.go exists
	if posErr != nil {
		// If position is invalid, we cannot proceed with AST-based analysis
		err := fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
		addAnalysisError(info, err, logger)
		// Attempt to gather package scope anyway, as it doesn't depend on cursor position
		gatherScopeContext(ctx, nil, targetPkg, fset, info, logger) // Pass nil path
		return err                                                  // Return error to indicate failure to get cursor context
	}
	info.CursorPos = cursorPos // Store valid cursor position
	logger = logger.With("cursorPos", info.CursorPos, "cursorPosStr", fset.PositionFor(info.CursorPos, true).String())
	logger.Debug("Calculated cursor position")

	// Proceed only if AST is available
	if targetFileAST != nil {
		// --- Call modular analysis functions ---
		// These functions populate the 'info' struct directly and add errors/diagnostics.

		// 1. Find enclosing path and identify specific context nodes (call, selector, etc.)
		path, pathErr := findEnclosingPathAndNodes(ctx, targetFileAST, info.CursorPos, targetPkg, fset, analyzer, info, logger)
		if pathErr != nil {
			// Error already added to info by the function
			logger.Warn("Error finding enclosing path or nodes, context may be less accurate", "error", pathErr)
			// Continue analysis if possible
		}

		// 2. Extract scope information based on path and package info
		scopeErr := extractScopeInformation(ctx, path, targetPkg, info.CursorPos, analyzer, info, logger)
		if scopeErr != nil {
			logger.Warn("Error extracting scope information", "error", scopeErr)
		}

		// 3. Extract relevant comments near the cursor
		commentErr := extractRelevantComments(ctx, targetFileAST, path, info.CursorPos, fset, analyzer, info, logger)
		if commentErr != nil {
			logger.Warn("Error extracting relevant comments", "error", commentErr)
		}

		// --- Add analysis-based diagnostics ---
		addAnalysisDiagnostics(fset, info, logger) // Assumes helpers_diagnostics.go exists

	} else {
		// AST is missing, likely due to load errors.
		addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil"), logger)
		// Attempt to gather package scope anyway
		gatherScopeContext(ctx, nil, targetPkg, fset, info, logger)
	}

	// Return nil, non-fatal errors are collected in info.AnalysisErrors
	return nil
}

// findEnclosingPathAndNodes finds the AST path and identifies context nodes.
// Populates info struct directly. Returns the path and any fatal error.
// ** MODIFIED: Cycle 3 - Pass context, ensure errors added correctly **
func findEnclosingPathAndNodes(
	ctx context.Context, // Added context
	targetFileAST *ast.File,
	cursorPos token.Pos,
	pkg *packages.Package,
	fset *token.FileSet,
	analyzer *GoPackagesAnalyzer, // For potential caching
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) ([]ast.Node, error) {
	// --- Memory Cache Check (Conceptual - Omitted for now) ---

	// --- Direct Implementation ---
	path, pathFindErr := findEnclosingPath(ctx, targetFileAST, cursorPos, info, logger)
	if pathFindErr != nil {
		// Error already added by findEnclosingPath
		return nil, pathFindErr // Return the error if path finding failed
	}
	// findContextNodes populates info directly and adds errors to info.AnalysisErrors
	findContextNodes(ctx, path, cursorPos, pkg, fset, analyzer, info, logger)
	// Return the path, non-fatal errors are collected in info.AnalysisErrors
	return path, nil
}

// extractScopeInformation gathers variables/types in scope.
// Populates info struct directly. Returns any fatal error.
// ** MODIFIED: Cycle 3 - Pass context **
func extractScopeInformation(
	ctx context.Context, // Added context
	path []ast.Node,
	targetPkg *packages.Package,
	cursorPos token.Pos,
	analyzer *GoPackagesAnalyzer, // For potential caching
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) error {
	// --- Memory Cache Check (Conceptual - Omitted for now) ---

	// --- Direct Implementation ---
	gatherScopeContext(ctx, path, targetPkg, info.TargetFileSet, info, logger) // Populates info, adds errors internally
	return nil                                                                 // Errors are added to info.AnalysisErrors internally
}

// extractRelevantComments finds comments near the cursor.
// Populates info struct directly. Returns any fatal error.
// ** MODIFIED: Cycle 3 - Pass context **
func extractRelevantComments(
	ctx context.Context, // Added context
	targetFileAST *ast.File,
	path []ast.Node,
	cursorPos token.Pos,
	fset *token.FileSet,
	analyzer *GoPackagesAnalyzer, // For potential caching
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) error {
	// --- Memory Cache Check (Conceptual - Omitted for now) ---

	// --- Direct Implementation ---
	findRelevantComments(ctx, targetFileAST, path, cursorPos, fset, info, logger) // Populates info, adds errors internally
	return nil                                                                    // Errors are added to info.AnalysisErrors internally
}

// findEnclosingPath finds the AST node path from the root to the node enclosing the cursor.
// ** MODIFIED: Cycle 3 - Pass context, return error **
func findEnclosingPath(ctx context.Context, targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo, logger *slog.Logger) ([]ast.Node, error) {
	if targetFileAST == nil {
		err := errors.New("cannot find enclosing path: targetFileAST is nil")
		addAnalysisError(info, err, logger)
		return nil, err
	}
	if !cursorPos.IsValid() {
		err := errors.New("cannot find enclosing path: invalid cursor position")
		addAnalysisError(info, err, logger)
		return nil, err
	}
	// astutil.PathEnclosingInterval finds the tightest enclosing node sequence
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		logger.Debug("No AST path found enclosing cursor position", "pos", cursorPos)
		// Not necessarily an error, could be whitespace
	}
	return path, nil
}

// gatherScopeContext walks the enclosing path to find relevant scope information.
// ** MODIFIED: Cycle 3 - Pass context, improve error handling/logging **
func gatherScopeContext(ctx context.Context, path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	addPackageScope(ctx, targetPkg, info, logger) // Add package scope first

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
							logger.Warn("Could not format receiver type", "error", err)
							info.ReceiverType = "[error formatting receiver]"
							addAnalysisError(info, fmt.Errorf("receiver format error: %w", err), logger)
						}
					} else if n.Recv != nil {
						logger.Debug("Could not get receiver type string from AST", "recv", n.Recv)
					}
				}
				// Try to get type info for the function
				funcName := "[anonymous]"
				if n.Name != nil {
					funcName = n.Name.Name
				}
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
					if obj, ok := targetPkg.TypesInfo.Defs[n.Name]; ok && obj != nil {
						if fn, ok := obj.(*types.Func); ok {
							if info.EnclosingFunc == nil { // Capture the first Func type object
								info.EnclosingFunc = fn
							}
							// Add signature parameters/results to the current scope map
							if sig, ok := fn.Type().(*types.Signature); ok {
								addSignatureToScope(sig, info.VariablesInScope)
							} else {
								logger.Warn("Function object type is not a signature", "func", funcName, "type", fn.Type())
							}
						} else {
							logger.Warn("Object defined for func name is not a *types.Func", "func", funcName, "object_type", fmt.Sprintf("%T", obj))
						}
					} else if info.EnclosingFunc == nil { // Only log error if we haven't found type info yet
						addAnalysisError(info, fmt.Errorf("definition for func '%s' not found in TypesInfo", funcName), logger)
					}
				} else if info.EnclosingFunc == nil { // Log if type info is missing and we need it
					reason := getMissingTypeInfoReason(targetPkg)
					addAnalysisError(info, fmt.Errorf("type info for enclosing func '%s' unavailable: %s", funcName, reason), logger)
				}

			case *ast.BlockStmt:
				// Add variables declared in this block
				if info.EnclosingBlock == nil {
					info.EnclosingBlock = n
				} // Capture the innermost block
				posStr := getPosString(fset, n.Pos()) // Use helper for safe position string
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						// Add variables declared *before* the cursor within this block
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					} else { // Scope info missing for this node
						addAnalysisError(info, fmt.Errorf("scope info missing for block at %s", posStr), logger)
					}
				} else if info.EnclosingBlock == n { // Log if type info is missing for the innermost block
					reason := getMissingTypeInfoReason(targetPkg)
					addAnalysisError(info, fmt.Errorf("cannot get scope variables for block at %s: %s", posStr, reason), logger)
				}
				// TODO: Add other scope-introducing nodes if necessary (e.g., *ast.IfStmt, *ast.ForStmt)
			}
		}
	} else {
		logger.Debug("AST path is nil, cannot gather block/function scopes.")
	}
}

// addPackageScope adds package-level identifiers to the scope map.
// ** MODIFIED: Cycle 3 - Pass context, improve error handling **
func addPackageScope(ctx context.Context, targetPkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
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
// ** MODIFIED: Cycle 3 - Pass context, ensure cmap creation is robust **
func findRelevantComments(ctx context.Context, targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if targetFileAST == nil || fset == nil {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"), logger)
		return
	}
	// Create a comment map for efficient lookup
	// Check for nil Comments field before creating map
	if targetFileAST.Comments == nil {
		logger.Debug("No comments found in target AST file.")
		info.CommentsNearCursor = []string{} // Ensure it's an empty slice, not nil
		return
	}
	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	if cmap == nil {
		// This might happen if fset or node is nil internally, though we checked targetFileAST
		addAnalysisError(info, errors.New("failed to create ast.CommentMap"), logger)
		info.CommentsNearCursor = []string{}
		return
	}
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset, logger)
}

// findCommentsWithMap implements the logic to find preceding or enclosing doc comments.
// ** MODIFIED: Cycle 3 - Improve preceding vs enclosing logic **
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, logger *slog.Logger) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		logger.Debug("Skipping comment finding due to nil cmap, invalid cursor, or nil fset")
		return comments
	}

	cursorPosInfo := fset.Position(cursorPos) // Get position info once
	cursorLine := cursorPosInfo.Line

	var precedingComments []string
	foundPrecedingOnLine := false
	minCommentPos := token.Pos(-1) // Track the closest preceding comment start

	// Strategy 1: Find comments immediately preceding the cursor line.
	// Iterate through all comment groups associated with any node in the file.
	for _, groups := range cmap { // Iterate over map values (comment groups for each node)
		if groups == nil {
			continue
		}
		for _, cg := range groups {
			if cg == nil || len(cg.List) == 0 {
				continue
			}

			// Check if the comment group ENDS on the line immediately before the cursor
			commentEndPosInfo := fset.Position(cg.End())
			if !commentEndPosInfo.IsValid() {
				continue // Skip invalid comment positions
			}
			commentEndLine := commentEndPosInfo.Line

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

// findContextNodes identifies specific AST nodes (call expr, selector expr, identifier)
// at or enclosing the cursor position and populates the AstContextInfo struct.
// It also attempts to resolve type information for these nodes using the provided package info.
// ** MODIFIED: Cycle 3 - Pass context, improve logging/error handling **
func findContextNodes(
	ctx context.Context, // Added context
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
		} else {
			logger.Debug("Cannot find context nodes: AST path is empty.")
		}
		return // Cannot determine context without path or fileset
	}

	posStr := func(p token.Pos) string { return getPosString(fset, p) } // Assumes helpers_diagnostics.go exists
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
		reason := getMissingTypeInfoReason(pkg) // Assumes helpers_diagnostics.go exists
		addAnalysisError(info, fmt.Errorf("cannot perform type analysis: %s", reason), logger)
	}

	// --- Check for Composite Literal ---
	innermostNode := path[0]
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
		litPosStr := posStr(compLit.Pos())
		logger.Debug("Cursor inside Composite Literal", "pos", litPosStr)
		info.CompositeLit = compLit
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[compLit]; ok {
				info.CompositeLitType = tv.Type
				if info.CompositeLitType == nil {
					addAnalysisError(info, fmt.Errorf("composite literal type resolved to nil at %s", litPosStr), logger)
				}
			} else {
				addAnalysisError(info, fmt.Errorf("missing type info for composite literal at %s", litPosStr), logger)
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
		callPosStr := posStr(callExpr.Pos())
		funPosStr := posStr(callExpr.Fun.Pos())
		logger.Debug("Cursor inside Call Expression arguments", "pos", callPosStr)
		info.CallExpr = callExpr
		info.CallArgIndex = calculateArgIndex(callExpr.Args, cursorPos)
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[callExpr.Fun]; ok && tv.Type != nil {
				if sig, ok := tv.Type.Underlying().(*types.Signature); ok {
					info.CallExprFuncType = sig
					info.ExpectedArgType = determineExpectedArgType(sig, info.CallArgIndex)
				} else {
					addAnalysisError(info, fmt.Errorf("type of call func (%T) at %s is %T, not signature", callExpr.Fun, funPosStr, tv.Type), logger)
				}
			} else {
				reason := "missing type info entry"
				if ok && tv.Type == nil {
					reason = "type info resolved to nil"
				}
				addAnalysisError(info, fmt.Errorf("%s for call func (%T) at %s", reason, callExpr.Fun, funPosStr), logger)
			}
		}
		return // Found context, exit
	}

	// --- Check for Selector Expression ---
	// Check first two nodes in path for selector, cursor must be *after* the dot
	for i := 0; i < len(path) && i < 2; i++ {
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok && cursorPos > selExpr.X.End() {
			selPosStr := posStr(selExpr.Pos())
			basePosStr := posStr(selExpr.X.Pos())
			logger.Debug("Cursor inside Selector Expression", "pos", selPosStr)
			info.SelectorExpr = selExpr
			if hasTypeInfo && typesMap != nil {
				if tv, ok := typesMap[selExpr.X]; ok {
					info.SelectorExprType = tv.Type
					if tv.Type == nil {
						addAnalysisError(info, fmt.Errorf("type info resolved to nil for selector base expr (%T) starting at %s", selExpr.X, basePosStr), logger)
					}
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info entry for selector base expr (%T) starting at %s", selExpr.X, basePosStr), logger)
				}
			}
			// Check if selected member is known (moved to addAnalysisDiagnostics)
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

	if ident != nil {
		identPosStr := posStr(ident.Pos())
		logger.Debug("Cursor at Identifier", "name", ident.Name, "pos", identPosStr)
		info.IdentifierAtCursor = ident
		if hasTypeInfo {
			var obj types.Object
			// Prefer 'Uses' map for identifiers that are not definitions
			if usesMap != nil {
				obj = usesMap[ident]
			}
			// Fallback to 'Defs' map if not found in 'Uses' (e.g., cursor is on the definition itself)
			if obj == nil && defsMap != nil {
				obj = defsMap[ident]
			}

			if obj != nil {
				info.IdentifierObject = obj
				info.IdentifierType = obj.Type()
				defPosStr := posStr(obj.Pos()) // Get pos string here for logging
				if info.IdentifierType == nil {
					addAnalysisError(info, fmt.Errorf("object '%s' at %s found but type is nil", obj.Name(), defPosStr), logger)
				}

				// --- Find Defining Node for Hover/Definition ---
				findDefiningNode(ctx, obj, fset, pkg, info, logger) // Populates info.IdentifierDefNode

			} else { // Object not found in defs or uses (diagnostic added elsewhere)
				// Try fallback to type map for the identifier itself
				if typesMap != nil {
					if tv, ok := typesMap[ident]; ok && tv.Type != nil {
						info.IdentifierType = tv.Type
					} else {
						// Only add error if object wasn't found AND type map failed
						addAnalysisError(info, fmt.Errorf("missing object and type info for identifier '%s' at %s", ident.Name, identPosStr), logger)
					}
				} else {
					addAnalysisError(info, fmt.Errorf("object not found for identifier '%s' at %s (and Types map is nil)", ident.Name, identPosStr), logger)
				}
			}
		} else {
			// Type info is missing, cannot resolve identifier object or type
			addAnalysisError(info, errors.New("missing type info for identifier analysis"), logger)
		}
		// No return here, allow other contexts to be checked if needed (though unlikely)
	}
	logger.Debug("Finished context node identification.")
}

// findDefiningNode attempts to find the AST node where a types.Object is defined.
// Populates info.IdentifierDefNode.
// ** MODIFIED: Cycle 3 - Pass context, improve error handling **
func findDefiningNode(ctx context.Context, obj types.Object, fset *token.FileSet, pkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	defPos := obj.Pos()
	if !defPos.IsValid() {
		logger.Debug("Object has invalid definition position, cannot find defining node.", "object", obj.Name())
		return
	}
	defPosStr := getPosString(fset, defPos) // Get pos string once

	// Find the token.File containing the definition
	defFile := fset.File(defPos)
	if defFile == nil {
		addAnalysisError(info, fmt.Errorf("could not find token.File for definition of '%s' at pos %s", obj.Name(), defPosStr), logger)
		return
	}
	defFileName := defFile.Name()
	logger = logger.With("def_file", defFileName)

	var defAST *ast.File
	// Check if definition is in the currently analyzed file
	if defFileName == info.FilePath {
		defAST = info.TargetAstFile
		logger.Debug("Definition is in the current file")
	} else {
		// Definition is in another file in the package. Find its AST.
		if pkg != nil && pkg.Syntax != nil {
			found := false
			for _, syntaxFile := range pkg.Syntax {
				// Check if the file object associated with the syntax file's position matches the definition file object
				if syntaxFile != nil && fset.File(syntaxFile.Pos()) == defFile {
					defAST = syntaxFile
					found = true
					logger.Debug("Found definition AST in package syntax")
					break
				}
			}
			if !found {
				logger.Warn("Definition file AST not found within package syntax", "package", pkg.PkgPath)
			}
		} else {
			reason := "package is nil"
			if pkg != nil {
				reason = "package.Syntax is nil"
			}
			logger.Warn("Cannot search for definition AST in other files", "reason", reason)
		}
	}

	if defAST == nil {
		addAnalysisError(info, fmt.Errorf("could not find AST for definition file '%s' of object '%s'", defFileName, obj.Name()), logger)
		return
	}

	// Find the specific AST node at the definition position
	defPath, _ := astutil.PathEnclosingInterval(defAST, defPos, defPos)
	if len(defPath) == 0 {
		addAnalysisError(info, fmt.Errorf("could not find AST path for definition of '%s' at pos %s", obj.Name(), defPosStr), logger)
		return
	}

	// Find the most specific node that corresponds to the definition
	for _, node := range defPath {
		isDeclNode := false
		switch n := node.(type) {
		case *ast.FuncDecl: // Function/Method declaration
			if n.Name != nil && n.Name.Pos() == defPos {
				isDeclNode = true
			}
		case *ast.ValueSpec: // Var or Const declaration (can be multiple names)
			for _, name := range n.Names {
				if name != nil && name.Pos() == defPos {
					isDeclNode = true
					break
				}
			}
		case *ast.TypeSpec: // Type declaration
			if n.Name != nil && n.Name.Pos() == defPos {
				isDeclNode = true
			}
		case *ast.Field: // Struct field, interface method, func param/result name
			for _, name := range n.Names {
				if name != nil && name.Pos() == defPos {
					isDeclNode = true
					break
				}
			}
		case *ast.AssignStmt: // Check for short variable declaration `:=`
			if n.Tok == token.DEFINE {
				for _, lhsExpr := range n.Lhs {
					if id, ok := lhsExpr.(*ast.Ident); ok && id.Pos() == defPos {
						isDeclNode = true
						break
					}
				}
			}
		}
		if isDeclNode {
			info.IdentifierDefNode = node
			logger.Debug("Found defining AST node for hover/definition", "object", obj.Name(), "node_type", fmt.Sprintf("%T", node), "pos", getPosString(fset, node.Pos()))
			return // Found the most specific declaration node
		}
	}

	// Could not find a specific decl node, maybe use innermost node at defPos?
	info.IdentifierDefNode = defPath[0]
	logger.Warn("Could not pinpoint specific defining declaration node, using innermost node at definition position", "object", obj.Name(), "innermost_type", fmt.Sprintf("%T", defPath[0]))
	// Don't add an error here, as we still found *a* node, just maybe not the best one.
	// addAnalysisError(info, fmt.Errorf("could not find specific defining node for '%s' at pos %s", obj.Name(), defPosStr), logger)
}

// calculateArgIndex determines the 0-based index of the argument the cursor is in.
// ** MODIFIED: Cycle 3 - Handle nil arguments **
func calculateArgIndex(args []ast.Expr, cursorPos token.Pos) int {
	if len(args) == 0 {
		return 0 // No args, cursor is effectively at index 0
	}
	for i, arg := range args {
		if arg == nil {
			// Handle nil argument - treat its span as minimal?
			// Or assume cursor cannot be "inside" a nil argument.
			// Let's assume cursor must be before or after the position where the nil arg would be.
			// If cursor is before the next non-nil arg's start, associate with this nil arg's index.
			if i+1 < len(args) && args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i
			}
			// Otherwise, continue checking subsequent args.
			continue
		}
		argStart := arg.Pos()
		argEnd := arg.End()
		// Determine the start of the "slot" for this argument, considering commas
		slotStart := argStart
		if i > 0 && args[i-1] != nil {
			// Slot starts after the previous argument's end (and potential comma/whitespace)
			slotStart = args[i-1].End() + 1
		} else if i > 0 {
			// Previous arg was nil, need to find the position of the comma before this arg
			// This is complex, fallback to argStart for now.
			slotStart = argStart
		}

		// Cursor is between start of this arg's slot and end of this arg
		if cursorPos >= slotStart && cursorPos <= argEnd {
			return i
		}
		// Cursor is after this arg's end position
		if cursorPos > argEnd {
			// If this is the last arg, cursor is positioned for the next potential arg
			if i == len(args)-1 {
				return i + 1
			}
			// If cursor is before the start of the *next* arg, it's in the slot for the next arg
			if args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i + 1
			}
			// Otherwise, cursor is somewhere between args[i] and args[i+1], likely after a comma.
			// We associate it with the *next* argument index.
		}
	}
	// Cursor is before the first argument's start position
	if len(args) > 0 && args[0] != nil && cursorPos < args[0].Pos() {
		return 0
	}
	// Default fallback (e.g., cursor after last arg's comma, or unexpected position)
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
		return nil // No parameters
	}

	if sig.Variadic() {
		if argIndex >= numParams-1 { // Cursor is at or after the variadic parameter
			lastParam := params.At(numParams - 1)
			if lastParam == nil {
				return nil
			}
			// The type of the variadic parameter itself is a slice (e.g., ...int is []int)
			// We expect the *element* type for individual arguments passed to it.
			if slice, ok := lastParam.Type().(*types.Slice); ok {
				return slice.Elem() // Expect element type
			}
			return nil // Should be a slice, but wasn't
		}
		// Cursor is before the variadic parameter
		param := params.At(argIndex)
		if param == nil {
			return nil
		}
		return param.Type()
	}

	// Not variadic
	if argIndex < numParams {
		param := params.At(argIndex)
		if param == nil {
			return nil
		}
		return param.Type()
	}

	return nil // Index out of bounds for non-variadic function
}

// listTypeMembers attempts to list exported fields and methods for a given type.
// ** MODIFIED: Cycle 3 - Add logging, handle nil gracefully **
func listTypeMembers(typ types.Type, expr ast.Expr, qualifier types.Qualifier, logger *slog.Logger) []MemberInfo {
	if logger == nil {
		logger = slog.Default() // Ensure logger is not nil
	}
	if typ == nil {
		logger.Debug("Cannot list members: input type is nil")
		return nil
	}
	logger = logger.With("type", typ.String())

	var members []MemberInfo
	seenMembers := make(map[string]MemberKind) // Track members to avoid duplicates from embedding/method sets

	// --- Get methods from the method set ---
	// Check both T and *T for methods
	msets := []*types.MethodSet{types.NewMethodSet(typ)}
	if _, isInterface := typ.Underlying().(*types.Interface); !isInterface {
		// If not an interface, also check pointer type for methods with pointer receivers
		if ptrType := types.NewPointer(typ); ptrType != nil {
			msets = append(msets, types.NewMethodSet(ptrType))
		}
	}

	for _, mset := range msets {
		if mset == nil {
			continue // Should not happen, but defensive check
		}
		for i := 0; i < mset.Len(); i++ {
			sel := mset.At(i)
			if sel == nil {
				continue
			}
			methodObj := sel.Obj()
			// Ensure it's an exported function/method
			if method, ok := methodObj.(*types.Func); ok && method != nil && method.Exported() {
				methodName := method.Name()
				// Add if not already seen (handles potential duplicates from T and *T method sets)
				if _, exists := seenMembers[methodName]; !exists {
					members = append(members, MemberInfo{Name: methodName, Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
					seenMembers[methodName] = MethodMember
					logger.Debug("Added method", "name", methodName)
				}
			}
		}
	}

	// --- Get fields from struct type (if applicable) ---
	// Need to handle pointer types correctly to get the underlying struct
	currentType := typ
	if ptr, ok := typ.(*types.Pointer); ok {
		if ptr.Elem() == nil {
			logger.Debug("Cannot list fields: pointer element type is nil")
			return members // Return methods found so far
		}
		currentType = ptr.Elem()
	}
	underlying := currentType.Underlying()
	if underlying == nil { // Handle cases where underlying type might be nil
		logger.Debug("Cannot list members: underlying type is nil")
		return members
	}

	if st, ok := underlying.(*types.Struct); ok {
		logger.Debug("Type is a struct, listing fields", "num_fields", st.NumFields())
		for i := 0; i < st.NumFields(); i++ {
			field := st.Field(i)
			// Add exported fields
			if field != nil && field.Exported() {
				fieldName := field.Name()
				// Add field only if a method with the same name hasn't already been added
				if _, exists := seenMembers[fieldName]; !exists {
					members = append(members, MemberInfo{Name: fieldName, Kind: FieldMember, TypeString: types.TypeString(field.Type(), qualifier)})
					seenMembers[fieldName] = FieldMember // Mark as seen (though unlikely to conflict here)
					logger.Debug("Added field", "name", fieldName)
				}
			}
			// TODO: Handle embedded fields recursively if needed?
			// This would involve checking field.Embedded() and recursively calling listTypeMembers
			// on the embedded type, potentially prefixing member names.
		}
	} else if iface, ok := underlying.(*types.Interface); ok {
		// Interfaces don't have fields, but list their explicit methods
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
		// TODO: Handle embedded interfaces?
	} else {
		logger.Debug("Type is not a struct or interface, no fields/explicit methods to list directly.", "type", fmt.Sprintf("%T", underlying))
	}

	logger.Debug("Finished listing members", "count", len(members))
	return members
}
