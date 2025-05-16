// deepcomplete/helpers_analysis_steps.go
// Contains helper functions for the core analysis steps (finding nodes, scope, comments).
package deepcomplete

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log/slog"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

// analysisCacheTTL is now derived from config within the analyzer (GoPackagesAnalyzer.getConfig().MemoryCacheTTL)

// ============================================================================
// Analysis Step Orchestration & Core Logic Helpers
// ============================================================================

// performAnalysisSteps orchestrates the detailed analysis after loading.
// It populates the AstContextInfo struct with findings and diagnostics.
// Returns an error only if a fatal error occurs preventing further analysis (e.g., invalid cursor).
// Non-fatal errors are added to info.AnalysisErrors.
// NOTE: This function is being refactored out. Logic is moving to specific Analyzer methods.
// Kept temporarily for compatibility with the main Analyze method.
func performAnalysisSteps(
	ctx context.Context,
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package,
	fset *token.FileSet,
	line, col int, // 1-based line/col from request
	analyzer Analyzer, // Analyzer interface
	info *AstContextInfo, // Pass info struct to be populated
	logger *slog.Logger,
) error {
	if fset == nil {
		err := errors.New("performAnalysisSteps requires non-nil fset")
		addAnalysisError(info, err, logger)
		return err // Fatal: fset is essential
	}
	if logger == nil {
		logger = slog.Default()
	}

	if targetFile == nil {
		logger.Warn("Target token.File is nil in performAnalysisSteps; file-specific analysis will be limited.")
		addAnalysisError(info, errors.New("target token.File is nil, cannot perform file-specific analysis"), logger)
		// Attempt to gather package scope only if targetFile is nil
		// This relies on GetScopeInfo which should handle nil targetFileAST gracefully if possible.
		scopeInfo, scopeErr := analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, line, col)
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information when targetFile is nil", "error", scopeErr)
			addAnalysisError(info, scopeErr, logger)
		} else if scopeInfo != nil {
			info.VariablesInScope = scopeInfo.Variables
		}
		return nil // Not a fatal error for performAnalysisSteps itself if targetFile is nil.
	}

	// Calculate cursor position (0-based token.Pos)
	cursorPos, posErr := calculateCursorPos(targetFile, line, col, logger)
	if posErr != nil {
		err := fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
		addAnalysisError(info, err, logger) // Add as non-fatal error to info
		// Attempt package scope analysis even with invalid cursor
		scopeInfo, scopeErr := analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, line, col)
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information after cursor error", "error", scopeErr)
			addAnalysisError(info, scopeErr, logger)
		} else if scopeInfo != nil {
			info.VariablesInScope = scopeInfo.Variables
		}
		return err // Return the fatal cursor position error
	}
	info.CursorPos = cursorPos // Store valid cursor position
	logger = logger.With("cursorPos", info.CursorPos, "cursorPosStr", fset.PositionFor(info.CursorPos, true).String())
	logger.Debug("Calculated cursor position for analysis steps")

	if targetFileAST != nil {
		// --- Get Enclosing Context (Function/Method, Block) ---
		enclosingCtx, encErr := analyzer.GetEnclosingContext(ctx, info.FilePath, info.Version, line, col)
		if encErr != nil {
			logger.Warn("Error getting enclosing context during analysis steps", "error", encErr)
			addAnalysisError(info, encErr, logger)
		} else if enclosingCtx != nil {
			info.EnclosingFunc = enclosingCtx.Func
			info.EnclosingFuncNode = enclosingCtx.FuncNode
			info.ReceiverType = enclosingCtx.Receiver
			info.EnclosingBlock = enclosingCtx.Block
		}

		// --- Get Scope Info (Variables, Types in scope) ---
		scopeInfo, scopeErr := analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, line, col)
		if scopeErr != nil {
			logger.Warn("Error getting scope info during analysis steps", "error", scopeErr)
			addAnalysisError(info, scopeErr, logger)
		} else if scopeInfo != nil {
			info.VariablesInScope = scopeInfo.Variables
		}

		// --- Get Relevant Comments near cursor ---
		comments, commentErr := analyzer.GetRelevantComments(ctx, info.FilePath, info.Version, line, col)
		if commentErr != nil {
			logger.Warn("Error getting relevant comments during analysis steps", "error", commentErr)
			addAnalysisError(info, commentErr, logger)
		}
		info.CommentsNearCursor = comments // Assign even if empty or if error occurred (partial results)

		// --- Find Specific Context Nodes (CallExpr, SelectorExpr, Identifier) ---
		// This requires the AST path from root to cursor.
		path, pathErr := findEnclosingPath(ctx, targetFileAST, cursorPos, info, logger) // info is passed for logging context
		if pathErr != nil {
			addAnalysisError(info, fmt.Errorf("finding enclosing AST path failed in analysis steps: %w", pathErr), logger)
		}
		// findContextNodes populates fields in 'info' like CallExpr, SelectorExpr, IdentifierAtCursor, etc.
		findContextNodes(ctx, path, cursorPos, targetPkg, fset, analyzer, info, logger)

		// --- Add Diagnostics based on analysis findings ---
		addAnalysisDiagnostics(fset, info, logger) // Populates info.Diagnostics

	} else {
		// targetFileAST is nil, meaning detailed AST-based analysis is not possible.
		addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil"), logger)
		// Attempt package scope analysis even without AST, if GetScopeInfo supports it.
		scopeInfo, scopeErr := analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, line, col)
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information when targetFileAST is nil", "error", scopeErr)
			addAnalysisError(info, scopeErr, logger)
		} else if scopeInfo != nil {
			info.VariablesInScope = scopeInfo.Variables
		}
	}

	// Non-fatal errors are added to info.AnalysisErrors.
	// Return nil unless a fatal error (like invalid cursor) occurred earlier.
	return nil
}

// findEnclosingPath finds the AST node path from the root to the node enclosing the cursor.
// It populates info.AnalysisErrors if issues occur.
func findEnclosingPath(ctx context.Context, targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo, logger *slog.Logger) ([]ast.Node, error) {
	if targetFileAST == nil {
		err := errors.New("cannot find enclosing path: targetFileAST is nil")
		// This function is a helper; errors should be returned to the caller (e.g., specific Get* method)
		// which can then decide to add it to AstContextInfo.AnalysisErrors or handle it as fatal.
		return nil, err
	}
	if !cursorPos.IsValid() {
		err := errors.New("cannot find enclosing path: invalid cursor position")
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		// This is not necessarily an error to add to AnalysisErrors,
		// it might just mean the cursor is in whitespace.
		logger.Debug("No AST path found enclosing cursor position", "pos", cursorPos)
	}
	return path, nil
}

// gatherScopeContext walks the enclosing path to find relevant scope information.
// It populates info.VariablesInScope and info.EnclosingFunc/Node.
// This is an internal helper, primarily called by GetScopeInfo's compute function.
func gatherScopeContext(ctx context.Context, path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if info.VariablesInScope == nil {
		info.VariablesInScope = make(map[string]types.Object)
	}

	// Add package-level scope first
	addPackageScope(ctx, targetPkg, info, logger) // Modifies info.VariablesInScope and info.AnalysisErrors

	// Traverse AST path for block and function scopes
	if path != nil {
		for i := len(path) - 1; i >= 0; i-- { // Iterate from outermost to innermost relevant node
			node := path[i]
			switch n := node.(type) {
			case *ast.File:
				// File scope is handled by addPackageScope via targetPkg.Types.Scope()
				continue
			case *ast.FuncDecl:
				// Set enclosing function if not already set by a more specific (inner) one
				if info.EnclosingFuncNode == nil {
					info.EnclosingFuncNode = n
					// Try to resolve types.Func from AST node
					if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
						if obj, ok := targetPkg.TypesInfo.Defs[n.Name].(*types.Func); ok {
							info.EnclosingFunc = obj
						}
					}
				}
				// Add function parameters and receivers to scope
				if n.Recv != nil && len(n.Recv.List) > 0 { // Receiver
					for _, field := range n.Recv.List {
						for _, name := range field.Names {
							if name != nil && name.Name != "" && name.Name != "_" {
								if obj, ok := targetPkg.TypesInfo.Defs[name].(*types.Var); ok {
									if _, exists := info.VariablesInScope[name.Name]; !exists {
										info.VariablesInScope[name.Name] = obj
									}
								}
							}
						}
					}
				}
				if n.Type != nil && n.Type.Params != nil { // Parameters
					for _, field := range n.Type.Params.List {
						for _, name := range field.Names {
							if name != nil && name.Name != "" && name.Name != "_" {
								if obj, ok := targetPkg.TypesInfo.Defs[name].(*types.Var); ok {
									if _, exists := info.VariablesInScope[name.Name]; !exists {
										info.VariablesInScope[name.Name] = obj
									}
								}
							}
						}
					}
				}

			case *ast.BlockStmt:
				// Set innermost block if not already set
				if info.EnclosingBlock == nil {
					info.EnclosingBlock = n
				}
				// Add variables declared in this block
				blockPosStr := getPosString(fset, n.Pos()) // Use fset from info
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope) // Modifies info.VariablesInScope
					} else {
						// Scope info might be missing for a block if type checking failed for it.
						addAnalysisError(info, fmt.Errorf("scope info missing for block at %s", blockPosStr), logger)
					}
				} else if info.EnclosingBlock == n { // Log only for the primary enclosing block if type info is generally missing
					reason := getMissingTypeInfoReason(targetPkg)
					// Avoid spamming if type info is broadly unavailable
					if !strings.Contains(errors.Join(info.AnalysisErrors...).Error(), "type info") {
						addAnalysisError(info, fmt.Errorf("cannot get scope variables for block at %s: %s", blockPosStr, reason), logger)
					}
				}
			}
		}

		// If EnclosingFunc was resolved, ensure its signature (params/results) is in scope
		if info.EnclosingFunc != nil {
			if sig, ok := info.EnclosingFunc.Type().(*types.Signature); ok {
				addSignatureToScope(sig, info.VariablesInScope) // Modifies info.VariablesInScope
			}
		} else if info.EnclosingFuncNode != nil && info.EnclosingFuncNode.Type != nil {
			// Fallback for AST-only if types.Func wasn't resolved but FuncDecl AST is available
			// This is less reliable as it doesn't use resolved types.Object
			if info.EnclosingFuncNode.Type.Params != nil {
				for _, field := range info.EnclosingFuncNode.Type.Params.List {
					for _, name := range field.Names {
						if name != nil && name.Name != "" && name.Name != "_" {
							if _, exists := info.VariablesInScope[name.Name]; !exists {
								// Cannot reliably add to scope without types.Object, log for now.
								logger.Debug("Skipping adding AST param to scope due to missing type info object", "param", name.Name)
							}
						}
					}
				}
			}
		}

	} else {
		logger.Debug("AST path is nil, cannot gather block/function specific scopes.")
	}
}

// addPackageScope adds package-level identifiers to the scope map.
// Modifies info.VariablesInScope and info.AnalysisErrors.
func addPackageScope(ctx context.Context, targetPkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			// Add all package level identifiers; cursorPos is not relevant for package scope.
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope)
		} else {
			addAnalysisError(info, fmt.Errorf("package scope missing for pkg %s", targetPkg.PkgPath), logger)
		}
	} else {
		reason := "targetPkg is nil"
		if targetPkg != nil { // targetPkg exists but targetPkg.Types is nil
			reason = fmt.Sprintf("package.Types field is nil for pkg %s", targetPkg.PkgPath)
		}
		addAnalysisError(info, fmt.Errorf("cannot add package scope: %s", reason), logger)
	}
}

// addScopeVariables adds identifiers from a types.Scope if declared before cursorPos.
// For package scope, cursorPos is NoPos, so all are included.
// Modifies scopeMap.
func addScopeVariables(typeScope *types.Scope, cursorPos token.Pos, scopeMap map[string]types.Object) {
	if typeScope == nil {
		return
	}
	for _, name := range typeScope.Names() {
		obj := typeScope.Lookup(name)
		if obj == nil {
			continue
		}
		// Include if:
		// 1. cursorPos is invalid (e.g., for package scope, include all)
		// 2. Object's position is invalid (e.g., built-ins, include all)
		// 3. Object is declared before the cursor
		include := !cursorPos.IsValid() || !obj.Pos().IsValid() || obj.Pos() < cursorPos

		if include {
			if _, exists := scopeMap[name]; !exists { // Add only if not already shadowed by a more local scope
				switch obj.(type) {
				// Filter to relevant object kinds
				case *types.Var, *types.Const, *types.TypeName, *types.Func, *types.PkgName, *types.Builtin, *types.Nil:
					scopeMap[name] = obj
				}
			}
		}
	}
}

// addSignatureToScope adds named parameters and results from a types.Signature to the scope map.
// Modifies scopeMap.
func addSignatureToScope(sig *types.Signature, scopeMap map[string]types.Object) {
	if sig == nil {
		return
	}
	addTupleToScope(sig.Params(), scopeMap)  // Add parameters
	addTupleToScope(sig.Results(), scopeMap) // Add named results
}

// addTupleToScope adds named variables from a types.Tuple (like params or results) to the scope map.
// Modifies scopeMap.
func addTupleToScope(tuple *types.Tuple, scopeMap map[string]types.Object) {
	if tuple == nil {
		return
	}
	for j := 0; j < tuple.Len(); j++ {
		v := tuple.At(j)                                   // v is a *types.Var
		if v != nil && v.Name() != "" && v.Name() != "_" { // Ensure it's a named variable (not blank)
			if _, exists := scopeMap[v.Name()]; !exists { // Add only if not shadowed
				scopeMap[v.Name()] = v
			}
		}
	}
}

// findRelevantComments uses ast.CommentMap to find comments near the cursor.
// Populates info.CommentsNearCursor and info.AnalysisErrors.
func findRelevantComments(ctx context.Context, targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if targetFileAST == nil || fset == nil {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"), logger)
		if info.CommentsNearCursor == nil { // Ensure it's initialized
			info.CommentsNearCursor = []string{}
		}
		return
	}
	if targetFileAST.Comments == nil {
		logger.Debug("No comments found in target AST file (targetFileAST.Comments is nil).")
		if info.CommentsNearCursor == nil {
			info.CommentsNearCursor = []string{}
		}
		return
	}

	// Create CommentMap from the AST file's comments
	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	if cmap == nil {
		addAnalysisError(info, errors.New("failed to create ast.CommentMap"), logger)
		if info.CommentsNearCursor == nil {
			info.CommentsNearCursor = []string{}
		}
		return
	}
	// Use the helper that takes CommentMap, path, and cursorPos
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset, logger)
}

// findCommentsWithMap implements the logic to find preceding or enclosing doc comments using CommentMap.
// This version is a helper and returns the comments directly.
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, logger *slog.Logger) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		logger.Debug("Skipping comment finding: cmap, cursorPos, or fset invalid/nil.")
		return comments // Return empty slice
	}

	cursorPosInfo := fset.Position(cursorPos)
	cursorLine := cursorPosInfo.Line
	var precedingComments []string // Store comments found on the line immediately preceding the cursor
	foundPrecedingOnLine := false
	minCommentPosOnPrecedingLine := token.Pos(-1) // To get the "last" comment block on the preceding line

	// Iterate through all comment groups in the file via the CommentMap
	for _, groups := range cmap { // cmap is map[ast.Node][]*ast.CommentGroup
		if groups == nil {
			continue
		}
		for _, cg := range groups { // cg is *ast.CommentGroup
			if cg == nil || len(cg.List) == 0 {
				continue
			}
			commentEndPosInfo := fset.Position(cg.End())
			if !commentEndPosInfo.IsValid() {
				continue
			}
			commentEndLine := commentEndPosInfo.Line

			// Check if the comment group ends on the line just before the cursor's line
			if commentEndLine == cursorLine-1 {
				// If this is the first comment block on the preceding line, or it's later than previously found ones
				if !foundPrecedingOnLine || cg.Pos() > minCommentPosOnPrecedingLine {
					precedingComments = nil // Reset to get only the last block
					for _, c := range cg.List {
						if c != nil {
							precedingComments = append(precedingComments, c.Text)
						}
					}
					minCommentPosOnPrecedingLine = cg.Pos()
					foundPrecedingOnLine = true
				}
			}
		}
	}

	if foundPrecedingOnLine {
		logger.Debug("Found comments on the line immediately preceding cursor", "count", len(precedingComments))
		comments = append(comments, precedingComments...)
	} else {
		// If no comments on preceding line, look for doc comments on enclosing AST nodes
		logger.Debug("No comments on preceding line, searching for doc comments on enclosing AST nodes.")
		if path != nil {
			for i := 0; i < len(path); i++ { // Iterate from innermost to outermost node in path
				node := path[i]
				var docComment *ast.CommentGroup
				// Check for .Doc field on common declaration nodes
				switch n := node.(type) {
				case *ast.FuncDecl:
					docComment = n.Doc
				case *ast.GenDecl: // For var, const, type declarations
					docComment = n.Doc
				case *ast.TypeSpec: // If path directly points to TypeSpec inside GenDecl
					docComment = n.Doc
				case *ast.ValueSpec: // If path directly points to ValueSpec inside GenDecl
					docComment = n.Doc
				case *ast.Field: // For struct fields or interface methods
					docComment = n.Doc
				}
				if docComment != nil {
					logger.Debug("Found doc comment on an enclosing AST node", "node_type", fmt.Sprintf("%T", node))
					for _, c := range docComment.List {
						if c != nil {
							comments = append(comments, c.Text)
						}
					}
					goto cleanupAndUnique // Found doc comment, use it and stop searching further up
				}
			}
		}
	}

cleanupAndUnique:
	// Remove duplicate comment lines if any were collected from multiple sources
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
	logger.Debug("Final relevant comments determined", "count", len(comments))
	return comments
}

// findContextNodes identifies specific AST nodes (call expr, selector expr, identifier)
// at or enclosing the cursor position and populates the AstContextInfo struct.
// This is a key function for understanding the immediate context around the cursor.
// It populates fields like info.CallExpr, info.SelectorExpr, info.IdentifierAtCursor, etc.
func findContextNodes(
	ctx context.Context,
	path []ast.Node, // AST path from root to cursor
	cursorPos token.Pos,
	pkg *packages.Package, // Resolved package info
	fset *token.FileSet, // FileSet for position interpretation
	analyzer Analyzer, // Analyzer for potential recursive calls or config access
	info *AstContextInfo, // Struct to populate with findings
	logger *slog.Logger,
) {
	if len(path) == 0 {
		logger.Debug("Cannot find context nodes: AST path is empty.")
		return
	}
	if fset == nil { // Essential for position information
		addAnalysisError(info, errors.New("Fset is nil in findContextNodes, cannot proceed"), logger)
		return
	}

	posStr := func(p token.Pos) string { return getPosString(fset, p) } // Helper for logging positions

	// Check for type information availability
	hasTypeInfo := pkg != nil && pkg.TypesInfo != nil
	var typesMap map[ast.Expr]types.TypeAndValue
	var defsMap map[*ast.Ident]types.Object
	var usesMap map[*ast.Ident]types.Object

	if hasTypeInfo {
		typesMap = pkg.TypesInfo.Types
		defsMap = pkg.TypesInfo.Defs
		usesMap = pkg.TypesInfo.Uses
		// Log if any essential type info maps are nil, as this indicates incomplete type checking
		if typesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Types' (pkg.TypesInfo.Types) is nil"), logger)
		}
		if defsMap == nil {
			addAnalysisError(info, errors.New("type info map 'Defs' (pkg.TypesInfo.Defs) is nil"), logger)
		}
		if usesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Uses' (pkg.TypesInfo.Uses) is nil"), logger)
		}
	} else {
		reason := getMissingTypeInfoReason(pkg) // Get a string explaining why type info might be missing
		addAnalysisError(info, fmt.Errorf("cannot perform full type analysis for context nodes: %s", reason), logger)
	}

	// --- Identify Composite Literal Context ---
	// Check if the innermost node containing the cursor is a composite literal.
	innermostNode := path[0]
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
		compLitPosStr := posStr(compLit.Pos())
		logger.Debug("Cursor is inside a Composite Literal", "pos", compLitPosStr)
		info.CompositeLit = compLit
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[compLit]; ok { // tv is types.TypeAndValue
				info.CompositeLitType = tv.Type
				if info.CompositeLitType == nil {
					addAnalysisError(info, fmt.Errorf("composite literal type resolved to nil at %s", compLitPosStr), logger)
				}
			} else {
				addAnalysisError(info, fmt.Errorf("missing type info for composite literal at %s", compLitPosStr), logger)
			}
		}
		return // If in composite lit, usually other contexts like call/selector are less relevant for this immediate node
	}

	// --- Identify Call Expression Context ---
	// Check path[0] (innermost) and path[1] (parent) for CallExpr.
	var callExpr *ast.CallExpr
	if ce, ok := path[0].(*ast.CallExpr); ok {
		callExpr = ce
	} else if len(path) > 1 { // Check parent if innermost isn't a CallExpr
		if ce, ok := path[1].(*ast.CallExpr); ok {
			callExpr = ce
		}
	}

	if callExpr != nil && cursorPos > callExpr.Lparen && cursorPos <= callExpr.Rparen {
		// Cursor is within the parentheses of a call expression (i.e., in the argument list).
		callPosStr := posStr(callExpr.Pos())
		funPosStr := posStr(callExpr.Fun.Pos()) // Position of the function being called
		logger.Debug("Cursor is inside Call Expression arguments", "call_pos", callPosStr, "func_pos", funPosStr)
		info.CallExpr = callExpr
		info.CallArgIndex = calculateArgIndex(callExpr.Args, cursorPos) // Determine which argument index cursor is at

		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[callExpr.Fun]; ok && tv.Type != nil {
				if sig, okSig := tv.Type.Underlying().(*types.Signature); okSig {
					info.CallExprFuncType = sig // Store resolved function signature
					info.ExpectedArgType = determineExpectedArgType(sig, info.CallArgIndex)
				} else {
					addAnalysisError(info, fmt.Errorf("type of call func (%T) at %s is %T, not a signature", callExpr.Fun, funPosStr, tv.Type), logger)
				}
			} else {
				reason := "missing type info entry"
				if tv, ok := typesMap[callExpr.Fun]; ok && tv.Type == nil { // Check if entry exists but type is nil
					reason = "type info resolved to nil"
				}
				addAnalysisError(info, fmt.Errorf("%s for call func (%T) at %s", reason, callExpr.Fun, funPosStr), logger)
			}
		}
		return // If in call expression args, this is the primary context.
	}

	// --- Identify Selector Expression Context ---
	// Check path[0] and path[1] for SelectorExpr (e.g., x.Y).
	// Cursor must be after the '.' (X.End()) to be considered part of the selector name.
	for i := 0; i < len(path) && i < 2; i++ { // Check current and parent node
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok && cursorPos > selExpr.X.End() {
			selPosStr := posStr(selExpr.Pos())
			basePosStr := posStr(selExpr.X.Pos()) // Position of the expression 'x' in 'x.Y'
			logger.Debug("Cursor is inside a Selector Expression (after '.')", "selector_pos", selPosStr, "base_expr_pos", basePosStr)
			info.SelectorExpr = selExpr
			if hasTypeInfo && typesMap != nil {
				if tv, ok := typesMap[selExpr.X]; ok { // Get type of 'x'
					info.SelectorExprType = tv.Type
					if tv.Type == nil {
						addAnalysisError(info, fmt.Errorf("type info resolved to nil for selector base expr (%T) starting at %s", selExpr.X, basePosStr), logger)
					}
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info entry for selector base expr (%T) starting at %s", selExpr.X, basePosStr), logger)
				}
			}
			return // If in selector, this is the primary context.
		}
	}

	// --- Identify Identifier Context ---
	// If not in a specific construct like CallExpr args or SelectorExpr, check if cursor is at an Identifier.
	var ident *ast.Ident
	if id, ok := path[0].(*ast.Ident); ok && cursorPos >= id.Pos() && cursorPos <= id.End() {
		// Ensure this identifier is not the 'Sel' part of a SelectorExpr already handled,
		// nor the 'Fun' part of a CallExpr if we are looking for standalone identifiers.
		isSelectorSel := false
		isCallFun := false
		if len(path) > 1 {
			if sel, okSel := path[1].(*ast.SelectorExpr); okSel && sel.Sel == id {
				isSelectorSel = true
			}
			if call, okCall := path[1].(*ast.CallExpr); okCall && call.Fun == id { // Simpler check: is it the Fun part?
				isCallFun = true
			} else if len(path) > 2 { // Check if id is Sel of Fun in Call (e.g. pkg.Func())
				if selAsFun, okSelFun := path[1].(*ast.SelectorExpr); okSelFun {
					if call, okCallParent := path[2].(*ast.CallExpr); okCallParent && call.Fun == selAsFun && selAsFun.Sel == id {
						isCallFun = true
					}
				}
			}
		}

		if !isSelectorSel && !isCallFun { // Only consider if it's a "standalone" identifier in this context
			ident = id
		} else {
			logger.Debug("Identifier is part of a selector or call func, already handled or not primary focus here.", "ident_name", id.Name)
		}
	}

	if ident != nil {
		identPosStr := posStr(ident.Pos())
		logger.Debug("Cursor is at an Identifier", "name", ident.Name, "pos", identPosStr)
		info.IdentifierAtCursor = ident
		if hasTypeInfo {
			var obj types.Object
			// Try to find the object in Uses map first (most common for identifiers not at definition)
			if usesMap != nil {
				obj = usesMap[ident]
			}
			// If not in Uses, check Defs (if cursor is at the definition site)
			if obj == nil && defsMap != nil {
				obj = defsMap[ident]
			}

			if obj != nil {
				info.IdentifierObject = obj
				info.IdentifierType = obj.Type() // Type of the resolved object
				defObjPosStr := posStr(obj.Pos())
				if info.IdentifierType == nil {
					// This can happen for unresolved identifiers or issues in type checking
					addAnalysisError(info, fmt.Errorf("object '%s' at %s found but its type is nil", obj.Name(), defObjPosStr), logger)
				}
				// Attempt to find the AST node where this object was defined (for hover/definition)
				findDefiningNode(ctx, obj, fset, pkg, info, logger) // Populates info.IdentifierDefNode
			} else {
				// Object not found in Uses or Defs. Try to get type from Types map as a fallback.
				if typesMap != nil {
					if tv, ok := typesMap[ident]; ok && tv.Type != nil {
						info.IdentifierType = tv.Type // Type of the expression (identifier) itself
						logger.Debug("Identifier object not in Uses/Defs, but type found in Types map.", "name", ident.Name, "type", tv.Type.String())
					} else {
						addAnalysisError(info, fmt.Errorf("missing object and type info for identifier '%s' at %s", ident.Name, identPosStr), logger)
					}
				} else {
					// Types map is also nil, no type info available.
					addAnalysisError(info, fmt.Errorf("object not found for identifier '%s' at %s (and Types map is nil)", ident.Name, identPosStr), logger)
				}
			}
		} else {
			// Type info is generally unavailable for the package.
			addAnalysisError(info, errors.New("missing general type information for identifier analysis"), logger)
		}
	}
	logger.Debug("Finished context node identification.")
}

// findDefiningNode attempts to find the AST node where a types.Object is defined.
// This is crucial for features like "go to definition" and providing documentation on hover.
// It populates info.IdentifierDefNode.
func findDefiningNode(ctx context.Context, obj types.Object, fset *token.FileSet, pkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	defPos := obj.Pos() // Get the definition position of the object
	if !defPos.IsValid() {
		logger.Debug("Object has an invalid definition position, cannot find defining AST node.", "object_name", obj.Name())
		return
	}

	defPosStr := getPosString(fset, defPos) // For logging
	defFileToken := fset.File(defPos)       // Get the token.File where the definition occurs
	if defFileToken == nil {
		addAnalysisError(info, fmt.Errorf("could not find token.File for definition of '%s' at pos %s", obj.Name(), defPosStr), logger)
		return
	}
	defFileName := defFileToken.Name() // Absolute path to the definition file
	logger = logger.With("def_file", defFileName)

	var defAST *ast.File // AST of the file where the definition is located

	// Check if the definition is in the currently analyzed file
	if defFileName == info.FilePath { // info.FilePath is the absFilename of the current document
		defAST = info.TargetAstFile
		logger.Debug("Definition of object is in the current file.", "object_name", obj.Name())
	} else {
		// Definition is in a different file. Search for its AST within the loaded package's syntax trees.
		if pkg != nil && pkg.Syntax != nil {
			found := false
			for _, syntaxFile := range pkg.Syntax { // pkg.Syntax is []*ast.File
				if syntaxFile != nil && fset.File(syntaxFile.Pos()) == defFileToken {
					defAST = syntaxFile
					found = true
					logger.Debug("Found definition AST in another file within the package's syntax trees.", "object_name", obj.Name())
					break
				}
			}
			if !found {
				logger.Warn("Definition file AST not found within the loaded package's syntax trees.", "package_id", pkg.ID, "object_name", obj.Name())
				// This can happen if the definition is in a file not covered by the packages.Load pattern,
				// or if it's in a different package not directly imported/analyzed.
			}
		} else {
			reason := "package is nil"
			if pkg != nil { // pkg exists but pkg.Syntax is nil
				reason = "package.Syntax is nil"
			}
			logger.Warn("Cannot search for definition AST in other files.", "reason", reason, "object_name", obj.Name())
		}
	}

	if defAST == nil {
		addAnalysisError(info, fmt.Errorf("could not find or load AST for definition file '%s' of object '%s'", defFileName, obj.Name()), logger)
		return
	}

	// Find the AST path to the definition position within the definition AST
	defPath, _ := astutil.PathEnclosingInterval(defAST, defPos, defPos)
	if len(defPath) == 0 {
		addAnalysisError(info, fmt.Errorf("could not find AST path for definition of '%s' at pos %s in file %s", obj.Name(), defPosStr, defFileName), logger)
		return
	}

	// Iterate through the path to find the most specific declaration node
	// (e.g., *ast.FuncDecl, *ast.ValueSpec) that matches the object's definition position.
	for _, node := range defPath {
		isDeclNode := false
		switch n := node.(type) {
		case *ast.FuncDecl: // Function or method declaration
			if n.Name != nil && n.Name.Pos() == defPos {
				isDeclNode = true
			}
		case *ast.ValueSpec: // Var or const declaration
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
		case *ast.Field: // Struct field, interface method, function parameter/result
			for _, name := range n.Names { // Fields can have multiple names (e.g. i, j int)
				if name != nil && name.Pos() == defPos {
					isDeclNode = true
					break
				}
			}
			// For interface methods or embedded types in fields that might not have names
			if len(n.Names) == 0 && n.Type != nil && n.Type.Pos() == defPos {
				isDeclNode = true
			}
		case *ast.AssignStmt: // Short variable declaration (:=)
			if n.Tok == token.DEFINE {
				for _, lhsExpr := range n.Lhs {
					if id, ok := lhsExpr.(*ast.Ident); ok && id.Pos() == defPos {
						isDeclNode = true
						break
					}
				}
			}
		case *ast.ImportSpec: // Import declaration
			if n.Path != nil && n.Path.Pos() == defPos { // Path is usually the definition site for PkgName
				isDeclNode = true
			} else if n.Name != nil && n.Name.Pos() == defPos { // Named import
				isDeclNode = true
			}
		}

		if isDeclNode {
			info.IdentifierDefNode = node // Store the found AST declaration node
			logger.Debug("Found defining AST node for object.", "object_name", obj.Name(), "node_type", fmt.Sprintf("%T", node), "node_pos", getPosString(fset, node.Pos()))
			return
		}
	}

	// If no specific declaration node was matched, use the innermost node at the definition position as a fallback.
	// This might happen for complex cases or if the object's Pos doesn't exactly match a specific sub-node's Pos.
	info.IdentifierDefNode = defPath[0]
	logger.Warn("Could not pinpoint specific defining declaration node, using innermost AST node at definition position as fallback.", "object_name", obj.Name(), "innermost_node_type", fmt.Sprintf("%T", defPath[0]))
}

// calculateArgIndex determines the 0-based index of the argument the cursor is in within a call expression.
// This helps in providing context-specific help like signature help.
func calculateArgIndex(args []ast.Expr, cursorPos token.Pos) int {
	if len(args) == 0 { // No arguments, cursor is effectively at index 0 (for a potential first arg)
		return 0
	}

	for i, arg := range args {
		if arg == nil { // Should not happen in a valid AST, but handle defensively
			// If current arg is nil, and cursor is before the next non-nil arg, assume cursor is for current nil arg's slot.
			if i+1 < len(args) && args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i
			}
			continue // Skip nil arg
		}

		argStart := arg.Pos()
		argEnd := arg.End()

		// Define the "slot" for the current argument.
		// The slot starts after the previous argument's end (or at current arg's start if first).
		slotStart := argStart
		if i > 0 { // If not the first argument
			prevArg := args[i-1]
			if prevArg != nil {
				slotStart = prevArg.End() + 1 // Start of slot is after previous arg's comma
			}
			// If prevArg was nil, slotStart remains argStart of current.
		}

		// If cursor is within the span of the current argument (inclusive of its start, exclusive of its end for some interpretations)
		// or more robustly, if cursor is between the start of this arg's slot and its actual end.
		if cursorPos >= slotStart && cursorPos <= argEnd {
			return i
		}

		// If cursor is after the current argument
		if cursorPos > argEnd {
			if i == len(args)-1 { // If this is the last argument, cursor is for a new argument after it.
				return i + 1
			}
			// If cursor is before the start of the next argument, it's for a new argument in this slot.
			if args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i + 1
			}
			// Otherwise, cursor is likely within or after the next argument, loop will handle.
		}
	}

	// If cursor is before the first argument.
	if len(args) > 0 && args[0] != nil && cursorPos < args[0].Pos() {
		return 0
	}

	// Default fallback: if cursor is after all arguments, it's for a new argument at the end.
	return len(args)
}

// determineExpectedArgType finds the expected type for a given argument index in a function signature.
// Useful for type checking hints or more intelligent completions.
func determineExpectedArgType(sig *types.Signature, argIndex int) types.Type {
	if sig == nil || argIndex < 0 {
		return nil
	}
	params := sig.Params() // Parameters tuple
	if params == nil {
		return nil
	}
	numParams := params.Len()
	if numParams == 0 { // Function takes no parameters
		return nil
	}

	if sig.Variadic() { // If the function is variadic (e.g., ...T)
		// The last parameter in the signature is the variadic one.
		if argIndex >= numParams-1 { // If cursor is at or after the variadic parameter
			lastParam := params.At(numParams - 1)
			if lastParam == nil {
				return nil
			}
			// The type of the variadic parameter is a slice (e.g., []T).
			// The expected type for individual variadic arguments is the element type of that slice (T).
			if slice, ok := lastParam.Type().(*types.Slice); ok {
				return slice.Elem()
			}
			return nil // Should be a slice, but handle defensively
		}
		// If before the variadic parameter, it's a regular parameter.
		param := params.At(argIndex)
		if param == nil {
			return nil
		}
		return param.Type()
	} else { // Not variadic
		if argIndex < numParams { // Cursor is within the bounds of defined parameters
			param := params.At(argIndex)
			if param == nil {
				return nil
			}
			return param.Type()
		}
		// Cursor is beyond the number of non-variadic parameters (e.g., too many arguments).
		return nil
	}
}

// listTypeMembers attempts to list exported fields and methods for a given type.
// This is used, for example, in selector expression context to suggest members.
func listTypeMembers(typ types.Type, expr ast.Expr, qualifier types.Qualifier, logger *slog.Logger) []MemberInfo {
	if logger == nil {
		logger = slog.Default()
	}
	if typ == nil {
		logger.Debug("Cannot list members: input type is nil")
		return nil
	}
	logger = logger.With("type_string", typ.String()) // Log the string representation of the type

	var members []MemberInfo
	seenMembers := make(map[string]MemberKind) // To avoid duplicates if type and *type have same member names

	// --- Collect Methods ---
	// Consider methods on both T and *T, as they are both accessible.
	msets := []*types.MethodSet{types.NewMethodSet(typ)} // Methods on T
	// If typ is not an interface and not already a pointer, get methods for *T as well.
	if _, isInterface := typ.Underlying().(*types.Interface); !isInterface {
		if _, isPointer := typ.(*types.Pointer); !isPointer {
			if ptrType := types.NewPointer(typ); ptrType != nil {
				msets = append(msets, types.NewMethodSet(ptrType)) // Methods on *T
			}
		}
	}

	for _, mset := range msets {
		if mset == nil {
			continue
		}
		for i := 0; i < mset.Len(); i++ {
			sel := mset.At(i) // sel is a *types.Selection (method or field promoted from embedded struct)
			if sel == nil {
				continue
			}
			methodObj := sel.Obj() // This is the actual *types.Func for a method
			if method, ok := methodObj.(*types.Func); ok && method != nil && method.Exported() {
				methodName := method.Name()
				if _, exists := seenMembers[methodName]; !exists { // Add only if not seen before
					members = append(members, MemberInfo{
						Name:       methodName,
						Kind:       MethodMember,
						TypeString: types.TypeString(method.Type(), qualifier), // Format method signature
					})
					seenMembers[methodName] = MethodMember
					logger.Debug("Added method to member list", "method_name", methodName)
				}
			}
		}
	}

	// --- Collect Fields (for structs) ---
	// Dereference pointer types to get to the underlying struct definition.
	currentType := typ
	if ptr, ok := typ.(*types.Pointer); ok {
		if ptr.Elem() == nil { // Pointer to an incomplete or invalid type
			logger.Debug("Cannot list fields: pointer element type is nil")
			return members // Return already collected methods
		}
		currentType = ptr.Elem().Underlying() // Get the actual type pointed to, and its underlying type
	} else {
		currentType = typ.Underlying() // Get underlying type for non-pointers
	}

	if currentType == nil {
		logger.Debug("Cannot list fields: underlying type is nil after potential dereference.")
		return members
	}

	if st, ok := currentType.(*types.Struct); ok { // Check if the underlying type is a struct
		logger.Debug("Type is a struct, listing its fields.", "num_fields_in_struct", st.NumFields())
		for i := 0; i < st.NumFields(); i++ {
			field := st.Field(i)                  // field is a *types.Var
			if field != nil && field.Exported() { // Only include exported fields
				fieldName := field.Name()
				if _, exists := seenMembers[fieldName]; !exists { // Add only if not seen (e.g. shadowed by a method)
					members = append(members, MemberInfo{
						Name:       fieldName,
						Kind:       FieldMember,
						TypeString: types.TypeString(field.Type(), qualifier), // Format field type
					})
					seenMembers[fieldName] = FieldMember
					logger.Debug("Added field to member list", "field_name", fieldName)
				}
			}
		}
	} else if iface, ok := currentType.(*types.Interface); ok {
		// For interfaces, methods are already handled by MethodSet.
		// Explicitly list methods if needed for clarity or if MethodSet missed something (unlikely for explicit methods).
		logger.Debug("Type is an interface. Explicit methods already covered by MethodSet.", "num_explicit_methods", iface.NumExplicitMethods())
		// No new members added here as MethodSet should cover them.
	} else {
		logger.Debug("Type is not a struct or interface, no direct fields to list.", "actual_type_kind", fmt.Sprintf("%T", currentType))
	}

	logger.Debug("Finished listing type members", "total_members_found", len(members))
	return members
}
