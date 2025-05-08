// deepcomplete/helpers_analysis_steps.go
// Contains helper functions for the core analysis steps (finding nodes, scope, comments).
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
	"strings"
	"time" // Added for cache TTL

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

const (
	// Default TTL for memory cache entries related to analysis steps
	analysisCacheTTL = 5 * time.Minute
)

// ============================================================================
// Analysis Step Orchestration & Core Logic Helpers
// ============================================================================

// performAnalysisSteps orchestrates the detailed analysis after loading.
// It populates the AstContextInfo struct with findings and diagnostics.
// Returns an error only if a fatal error occurs preventing further analysis (e.g., invalid cursor).
// Non-fatal errors are added to info.AnalysisErrors.
func performAnalysisSteps(
	ctx context.Context, // Pass context
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package,
	fset *token.FileSet,
	line, col int, // 1-based line/col from request
	analyzer Analyzer, // Use interface type
	info *AstContextInfo, // Pass info struct to be populated
	logger *slog.Logger,
) error {
	if fset == nil {
		err := errors.New("performAnalysisSteps requires non-nil fset")
		addAnalysisError(info, err, logger) // Pass logger
		return err
	}
	if logger == nil {
		logger = slog.Default()
	}

	if targetFile == nil {
		logger.Warn("Target token.File is nil, proceeding with package scope analysis only.")
		addAnalysisError(info, errors.New("target token.File is nil, cannot perform file-specific analysis"), logger) // Pass logger
		// Pass ctx, logger, analyzer interface
		scopeErr := extractScopeInformation(ctx, nil, targetPkg, token.NoPos, analyzer, info, logger)
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information", "error", scopeErr)
		}
		return nil
	}

	// Calculate cursor position first (0-based token.Pos), pass logger
	cursorPos, posErr := calculateCursorPos(targetFile, line, col, logger) // Pass logger
	if posErr != nil {
		err := fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
		addAnalysisError(info, err, logger) // Pass logger
		// Attempt package scope analysis even with invalid cursor
		scopeErr := extractScopeInformation(ctx, nil, targetPkg, token.NoPos, analyzer, info, logger) // Pass ctx, logger, analyzer interface
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information after cursor error", "error", scopeErr)
		}
		return err // Return the original cursor position error
	}
	info.CursorPos = cursorPos
	logger = logger.With("cursorPos", info.CursorPos, "cursorPosStr", fset.PositionFor(info.CursorPos, true).String())
	logger.Debug("Calculated cursor position")

	if targetFileAST != nil {
		// Pass down analyzer interface, context, and logger
		// This function now uses caching internally
		path, pathErr := findEnclosingPathAndNodes(ctx, targetFileAST, info.CursorPos, targetPkg, fset, analyzer, info, logger)
		if pathErr != nil {
			logger.Warn("Error finding enclosing path or nodes, context may be less accurate", "error", pathErr)
			// Continue analysis if possible, pathErr is already added to info.AnalysisErrors
		}

		// Pass down analyzer interface, context, and logger
		// This function now uses caching internally
		scopeErr := extractScopeInformation(ctx, path, targetPkg, info.CursorPos, analyzer, info, logger)
		if scopeErr != nil {
			logger.Warn("Error extracting scope information", "error", scopeErr)
			// Error is added to info.AnalysisErrors internally
		}

		// Pass down analyzer interface, context, and logger
		// This function now uses caching internally
		commentErr := extractRelevantComments(ctx, targetFileAST, path, info.CursorPos, fset, analyzer, info, logger)
		if commentErr != nil {
			logger.Warn("Error extracting relevant comments", "error", commentErr)
			// Error is added to info.AnalysisErrors internally
		}

		// Pass logger
		addAnalysisDiagnostics(fset, info, logger) // Pass logger

	} else {
		addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil"), logger) // Pass logger
		// Attempt package scope analysis even without AST
		scopeErr := extractScopeInformation(ctx, nil, targetPkg, token.NoPos, analyzer, info, logger) // Pass ctx, logger, analyzer interface
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information without AST", "error", scopeErr)
		}
	}

	return nil
}

// findEnclosingPathAndNodes finds the AST path and identifies context nodes.
// Populates info struct directly. Returns the path and any fatal error.
// Uses memory cache for the path finding part.
func findEnclosingPathAndNodes(
	ctx context.Context, // Pass context
	targetFileAST *ast.File,
	cursorPos token.Pos,
	pkg *packages.Package,
	fset *token.FileSet,
	analyzer Analyzer, // Use interface type
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) ([]ast.Node, error) {
	if logger == nil {
		logger = slog.Default()
	}
	pathLogger := logger.With("op", "findEnclosingPathAndNodes")

	// --- Cache Path Finding ---
	cacheKey := generateCacheKey("astPath", info) // Use helper from helpers_cache.go
	computePathFn := func() ([]ast.Node, error) {
		return findEnclosingPath(ctx, targetFileAST, cursorPos, info, pathLogger) // Pass ctx, logger
	}

	// Cache the path ([]ast.Node). Use cost 1 as size is hard to determine easily.
	path, cacheHit, pathErr := withMemoryCache[[]ast.Node](analyzer, cacheKey, 1, analysisCacheTTL, computePathFn, pathLogger)
	if pathErr != nil {
		// Error already added by findEnclosingPath if it occurred there
		return nil, pathErr // Return the error if path finding failed
	}
	if cacheHit {
		pathLogger.Debug("AST path cache hit")
	}

	// --- Find Context Nodes (Not Cached Here, depends on path) ---
	// This part still runs even on cache hit for the path, as it populates info.
	findContextNodes(ctx, path, cursorPos, pkg, fset, analyzer, info, pathLogger) // Pass ctx, logger, analyzer interface

	// Return the path (cached or computed), non-fatal errors are collected in info.AnalysisErrors
	return path, nil
}

// extractScopeInformation gathers variables/types in scope.
// Populates info struct directly. Returns any fatal error.
// Uses memory cache.
func extractScopeInformation(
	ctx context.Context, // Pass context
	path []ast.Node,
	targetPkg *packages.Package,
	cursorPos token.Pos,
	analyzer Analyzer, // Use interface type
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) error {
	if logger == nil {
		logger = slog.Default()
	}
	scopeLogger := logger.With("op", "extractScopeInformation")

	// --- Cache Scope Extraction ---
	// Cache key depends on cursor position as scope varies within blocks/functions
	cacheKey := generateCacheKey("scopeInfo", info)
	computeScopeFn := func() (map[string]types.Object, error) {
		// Create a temporary map to store results, avoiding direct modification of info inside computeFn
		tempScopeMap := make(map[string]types.Object)
		// Create a temporary info struct copy for scope gathering to avoid race conditions if computeFn runs concurrently
		tempInfo := *info
		tempInfo.VariablesInScope = tempScopeMap // Point tempInfo to the temp map

		// gatherScopeContext populates tempInfo.VariablesInScope and adds errors to tempInfo.AnalysisErrors
		gatherScopeContext(ctx, path, targetPkg, info.TargetFileSet, &tempInfo, scopeLogger) // Pass ctx, logger

		// Check for errors added during scope gathering
		if len(tempInfo.AnalysisErrors) > len(info.AnalysisErrors) {
			// Collect only the *new* errors added by gatherScopeContext
			newErrors := tempInfo.AnalysisErrors[len(info.AnalysisErrors):]
			// Return the computed map and a wrapped error containing the new issues
			// Note: This error won't be cached by withMemoryCache
			return tempScopeMap, fmt.Errorf("errors during scope computation: %w", errors.Join(newErrors...))
		}
		// Return the computed scope map
		return tempScopeMap, nil
	}

	// Cache the scope map. Estimate cost based on number of items (simple approach).
	// TTL set by constant.
	computedScope, cacheHit, scopeErr := withMemoryCache[map[string]types.Object](
		analyzer, cacheKey, int64(len(info.VariablesInScope)+1), analysisCacheTTL, computeScopeFn, scopeLogger,
	)

	// If there was an error during computation (not a cache error), add it to the main info struct
	if scopeErr != nil {
		addAnalysisError(info, scopeErr, scopeLogger)
		// Continue even if scope computation had errors, might have partial results
	}

	// Update the main info struct with the cached or computed scope
	if computedScope != nil {
		info.VariablesInScope = computedScope
	}
	if cacheHit {
		scopeLogger.Debug("Scope information cache hit")
	}

	// gatherScopeContext adds errors to info internally, so we just return nil here.
	// Any fatal error would have been returned by withMemoryCache if computeFn returned it.
	return nil
}

// extractRelevantComments finds comments near the cursor.
// Populates info struct directly. Returns any fatal error.
// Uses memory cache.
func extractRelevantComments(
	ctx context.Context, // Pass context
	targetFileAST *ast.File,
	path []ast.Node,
	cursorPos token.Pos,
	fset *token.FileSet,
	analyzer Analyzer, // Use interface type
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) error {
	if logger == nil {
		logger = slog.Default()
	}
	commentLogger := logger.With("op", "extractRelevantComments")

	// --- Cache Comment Extraction ---
	cacheKey := generateCacheKey("comments", info)
	computeCommentsFn := func() ([]string, error) {
		// Create a temporary info struct copy to avoid race conditions
		tempInfo := *info
		tempInfo.CommentsNearCursor = nil // Start fresh

		// findRelevantComments populates tempInfo.CommentsNearCursor and adds errors to tempInfo.AnalysisErrors
		findRelevantComments(ctx, targetFileAST, path, cursorPos, fset, &tempInfo, commentLogger) // Pass ctx, logger

		// Check for errors added during comment finding
		if len(tempInfo.AnalysisErrors) > len(info.AnalysisErrors) {
			newErrors := tempInfo.AnalysisErrors[len(info.AnalysisErrors):]
			return tempInfo.CommentsNearCursor, fmt.Errorf("errors during comment computation: %w", errors.Join(newErrors...))
		}
		return tempInfo.CommentsNearCursor, nil
	}

	// Cache the resulting slice of strings. Estimate cost based on total length.
	computedComments, cacheHit, commentErr := withMemoryCache[[]string](
		analyzer, cacheKey, estimateCost(info.CommentsNearCursor), analysisCacheTTL, computeCommentsFn, commentLogger,
	)

	// If there was an error during computation, add it to the main info struct
	if commentErr != nil {
		addAnalysisError(info, commentErr, commentLogger)
		// Continue even if comment computation had errors
	}

	// Update the main info struct with the cached or computed comments
	if computedComments != nil {
		info.CommentsNearCursor = computedComments
	} else {
		info.CommentsNearCursor = []string{} // Ensure it's an empty slice if nil was returned/cached
	}

	if cacheHit {
		commentLogger.Debug("Relevant comments cache hit")
	}

	// findRelevantComments adds errors internally.
	return nil
}

// findEnclosingPath finds the AST node path from the root to the node enclosing the cursor.
// This is the actual computation function called by the cached wrapper.
func findEnclosingPath(ctx context.Context, targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo, logger *slog.Logger) ([]ast.Node, error) {
	if targetFileAST == nil {
		err := errors.New("cannot find enclosing path: targetFileAST is nil")
		addAnalysisError(info, err, logger) // Pass logger
		return nil, err
	}
	if !cursorPos.IsValid() {
		err := errors.New("cannot find enclosing path: invalid cursor position")
		addAnalysisError(info, err, logger) // Pass logger
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		logger.Debug("No AST path found enclosing cursor position", "pos", cursorPos)
	}
	return path, nil
}

// gatherScopeContext walks the enclosing path to find relevant scope information.
// This is the actual computation function called by the cached wrapper.
// It populates the provided info struct's VariablesInScope map.
func gatherScopeContext(ctx context.Context, path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	// Ensure the map exists before adding to it
	if info.VariablesInScope == nil {
		info.VariablesInScope = make(map[string]types.Object)
	}

	addPackageScope(ctx, targetPkg, info, logger) // Pass ctx, logger

	if path != nil {
		for i := len(path) - 1; i >= 0; i-- {
			node := path[i]
			switch n := node.(type) {
			case *ast.File:
				continue
			case *ast.FuncDecl:
				// Only capture the *first* FuncDecl encountered walking up
				if info.EnclosingFuncNode == nil {
					info.EnclosingFuncNode = n
					if fset != nil && n.Recv != nil && len(n.Recv.List) > 0 && n.Recv.List[0].Type != nil {
						var buf bytes.Buffer
						if err := format.Node(&buf, fset, n.Recv.List[0].Type); err == nil {
							info.ReceiverType = buf.String()
						} else {
							logger.Warn("Could not format receiver type", "error", err)
							info.ReceiverType = "[error formatting receiver]"
							addAnalysisError(info, fmt.Errorf("receiver format error: %w", err), logger) // Pass logger
						}
					} else if n.Recv != nil {
						logger.Debug("Could not get receiver type string from AST", "recv", n.Recv)
					}
				}
				funcName := "[anonymous]"
				if n.Name != nil {
					funcName = n.Name.Name
				}
				// Try to get type info only for the first encountered FuncDecl
				if info.EnclosingFunc == nil && targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
					if obj, ok := targetPkg.TypesInfo.Defs[n.Name]; ok && obj != nil {
						if fn, ok := obj.(*types.Func); ok {
							info.EnclosingFunc = fn // Capture the type info
							if sig, ok := fn.Type().(*types.Signature); ok {
								addSignatureToScope(sig, info.VariablesInScope)
							} else {
								logger.Warn("Function object type is not a signature", "func", funcName, "type", fn.Type())
							}
						} else {
							logger.Warn("Object defined for func name is not a *types.Func", "func", funcName, "object_type", fmt.Sprintf("%T", obj))
						}
					} else { // Def not found for this func name
						addAnalysisError(info, fmt.Errorf("definition for func '%s' not found in TypesInfo", funcName), logger) // Pass logger
					}
				} else if info.EnclosingFunc == nil { // Type info missing or already logged error
					reason := getMissingTypeInfoReason(targetPkg)
					// Avoid adding duplicate errors if type info is generally missing
					if !strings.Contains(errors.Join(info.AnalysisErrors...).Error(), "type info") {
						addAnalysisError(info, fmt.Errorf("type info for enclosing func '%s' unavailable: %s", funcName, reason), logger) // Pass logger
					}
				}

			case *ast.BlockStmt:
				if info.EnclosingBlock == nil { // Capture innermost block
					info.EnclosingBlock = n
				}
				posStr := getPosString(fset, n.Pos()) // Use helper from utils.go
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					} else { // Scope info missing for this specific block
						addAnalysisError(info, fmt.Errorf("scope info missing for block at %s", posStr), logger) // Pass logger
					}
				} else if info.EnclosingBlock == n { // Log only if type info is missing for the innermost block
					reason := getMissingTypeInfoReason(targetPkg)
					if !strings.Contains(errors.Join(info.AnalysisErrors...).Error(), "type info") {
						addAnalysisError(info, fmt.Errorf("cannot get scope variables for block at %s: %s", posStr, reason), logger) // Pass logger
					}
				}
			}
		}
	} else {
		logger.Debug("AST path is nil, cannot gather block/function scopes.")
	}
}

// addPackageScope adds package-level identifiers to the scope map.
func addPackageScope(ctx context.Context, targetPkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope) // cursorPos doesn't apply
		} else {
			addAnalysisError(info, fmt.Errorf("package scope missing for pkg %s", targetPkg.PkgPath), logger) // Pass logger
		}
	} else {
		reason := "targetPkg is nil"
		if targetPkg != nil {
			reason = fmt.Sprintf("package.Types field is nil for pkg %s", targetPkg.PkgPath)
		}
		addAnalysisError(info, fmt.Errorf("cannot add package scope: %s", reason), logger) // Pass logger
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
		include := !cursorPos.IsValid() || !obj.Pos().IsValid() || obj.Pos() < cursorPos
		if include {
			if _, exists := scopeMap[name]; !exists {
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
		if v != nil && v.Name() != "" {
			if _, exists := scopeMap[v.Name()]; !exists {
				scopeMap[v.Name()] = v
			}
		}
	}
}

// findRelevantComments uses ast.CommentMap to find comments near the cursor.
// This is the actual computation function called by the cached wrapper.
// It populates the provided info struct's CommentsNearCursor slice.
func findRelevantComments(ctx context.Context, targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if targetFileAST == nil || fset == nil {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"), logger) // Pass logger
		return
	}
	if targetFileAST.Comments == nil {
		logger.Debug("No comments found in target AST file.")
		if info.CommentsNearCursor == nil { // Initialize if nil
			info.CommentsNearCursor = []string{}
		}
		return
	}
	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	if cmap == nil {
		addAnalysisError(info, errors.New("failed to create ast.CommentMap"), logger) // Pass logger
		if info.CommentsNearCursor == nil {
			info.CommentsNearCursor = []string{}
		}
		return
	}
	// findCommentsWithMap returns the comments; assign to info struct
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset, logger) // Pass logger
}

// findCommentsWithMap implements the logic to find preceding or enclosing doc comments.
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, logger *slog.Logger) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		logger.Debug("Skipping comment finding due to nil cmap, invalid cursor, or nil fset")
		return comments
	}

	cursorPosInfo := fset.Position(cursorPos)
	cursorLine := cursorPosInfo.Line

	var precedingComments []string
	foundPrecedingOnLine := false
	minCommentPos := token.Pos(-1)

	for _, groups := range cmap {
		if groups == nil {
			continue
		}
		for _, cg := range groups {
			if cg == nil || len(cg.List) == 0 {
				continue
			}

			commentEndPosInfo := fset.Position(cg.End())
			if !commentEndPosInfo.IsValid() {
				continue
			}
			commentEndLine := commentEndPosInfo.Line

			if commentEndLine == cursorLine-1 {
				if !foundPrecedingOnLine || cg.Pos() > minCommentPos {
					precedingComments = nil
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

	if foundPrecedingOnLine {
		logger.Debug("Found preceding comments on line before cursor", "count", len(precedingComments))
		comments = append(comments, precedingComments...)
	} else {
		logger.Debug("No comments found on preceding line, looking for enclosing doc comments.")
		if path != nil {
			for i := 0; i < len(path); i++ {
				node := path[i]
				var docComment *ast.CommentGroup
				switch n := node.(type) {
				case *ast.FuncDecl:
					docComment = n.Doc
				case *ast.GenDecl:
					docComment = n.Doc
				case *ast.TypeSpec:
					docComment = n.Doc
				case *ast.Field:
					docComment = n.Doc
				case *ast.ValueSpec:
					docComment = n.Doc
				}
				if docComment != nil {
					logger.Debug("Found doc comment on enclosing node", "node_type", fmt.Sprintf("%T", node))
					for _, c := range docComment.List {
						if c != nil {
							comments = append(comments, c.Text)
						}
					}
					goto cleanup
				}
			}
		}
	}

cleanup:
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
// It also attempts to resolve type information for these nodes.
func findContextNodes(
	ctx context.Context, // Pass context
	path []ast.Node,
	cursorPos token.Pos,
	pkg *packages.Package,
	fset *token.FileSet,
	analyzer Analyzer, // Use interface type
	info *AstContextInfo, // Populate this struct
	logger *slog.Logger,
) {
	if len(path) == 0 || fset == nil {
		if fset == nil {
			addAnalysisError(info, errors.New("Fset is nil in findContextNodes"), logger) // Pass logger
		} else {
			logger.Debug("Cannot find context nodes: AST path is empty.")
		}
		return
	}

	posStr := func(p token.Pos) string { return getPosString(fset, p) } // Util func
	hasTypeInfo := pkg != nil && pkg.TypesInfo != nil
	var typesMap map[ast.Expr]types.TypeAndValue
	var defsMap map[*ast.Ident]types.Object
	var usesMap map[*ast.Ident]types.Object
	if hasTypeInfo {
		typesMap = pkg.TypesInfo.Types
		defsMap = pkg.TypesInfo.Defs
		usesMap = pkg.TypesInfo.Uses
		if typesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Types' is nil"), logger) // Pass logger
		}
		if defsMap == nil {
			addAnalysisError(info, errors.New("type info map 'Defs' is nil"), logger) // Pass logger
		}
		if usesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Uses' is nil"), logger) // Pass logger
		}
	} else {
		reason := getMissingTypeInfoReason(pkg)                                                // Assumes helpers_diagnostics.go exists
		addAnalysisError(info, fmt.Errorf("cannot perform type analysis: %s", reason), logger) // Pass logger
	}

	innermostNode := path[0]
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
		litPosStr := posStr(compLit.Pos())
		logger.Debug("Cursor inside Composite Literal", "pos", litPosStr)
		info.CompositeLit = compLit
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[compLit]; ok {
				info.CompositeLitType = tv.Type
				if info.CompositeLitType == nil {
					addAnalysisError(info, fmt.Errorf("composite literal type resolved to nil at %s", litPosStr), logger) // Pass logger
				}
			} else {
				addAnalysisError(info, fmt.Errorf("missing type info for composite literal at %s", litPosStr), logger) // Pass logger
			}
		}
		return
	}

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
					addAnalysisError(info, fmt.Errorf("type of call func (%T) at %s is %T, not signature", callExpr.Fun, funPosStr, tv.Type), logger) // Pass logger
				}
			} else {
				reason := "missing type info entry"
				if ok && tv.Type == nil {
					reason = "type info resolved to nil"
				}
				addAnalysisError(info, fmt.Errorf("%s for call func (%T) at %s", reason, callExpr.Fun, funPosStr), logger) // Pass logger
			}
		}
		return
	}

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
						addAnalysisError(info, fmt.Errorf("type info resolved to nil for selector base expr (%T) starting at %s", selExpr.X, basePosStr), logger) // Pass logger
					}
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info entry for selector base expr (%T) starting at %s", selExpr.X, basePosStr), logger) // Pass logger
				}
			}
			return
		}
	}

	var ident *ast.Ident
	if id, ok := path[0].(*ast.Ident); ok && cursorPos >= id.Pos() && cursorPos <= id.End() {
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
			if usesMap != nil {
				obj = usesMap[ident]
			}
			if obj == nil && defsMap != nil {
				obj = defsMap[ident]
			}

			if obj != nil {
				info.IdentifierObject = obj
				info.IdentifierType = obj.Type()
				defPosStr := posStr(obj.Pos())
				if info.IdentifierType == nil {
					addAnalysisError(info, fmt.Errorf("object '%s' at %s found but type is nil", obj.Name(), defPosStr), logger) // Pass logger
				}
				findDefiningNode(ctx, obj, fset, pkg, info, logger) // Pass ctx, logger

			} else {
				if typesMap != nil {
					if tv, ok := typesMap[ident]; ok && tv.Type != nil {
						info.IdentifierType = tv.Type
					} else {
						addAnalysisError(info, fmt.Errorf("missing object and type info for identifier '%s' at %s", ident.Name, identPosStr), logger) // Pass logger
					}
				} else {
					addAnalysisError(info, fmt.Errorf("object not found for identifier '%s' at %s (and Types map is nil)", ident.Name, identPosStr), logger) // Pass logger
				}
			}
		} else {
			addAnalysisError(info, errors.New("missing type info for identifier analysis"), logger) // Pass logger
		}
	}
	logger.Debug("Finished context node identification.")
}

// findDefiningNode attempts to find the AST node where a types.Object is defined.
// Populates info.IdentifierDefNode.
func findDefiningNode(ctx context.Context, obj types.Object, fset *token.FileSet, pkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	defPos := obj.Pos()
	if !defPos.IsValid() {
		logger.Debug("Object has invalid definition position, cannot find defining node.", "object", obj.Name())
		return
	}
	defPosStr := getPosString(fset, defPos) // Util func

	defFile := fset.File(defPos)
	if defFile == nil {
		addAnalysisError(info, fmt.Errorf("could not find token.File for definition of '%s' at pos %s", obj.Name(), defPosStr), logger) // Pass logger
		return
	}
	defFileName := defFile.Name()
	logger = logger.With("def_file", defFileName)

	var defAST *ast.File
	if defFileName == info.FilePath {
		defAST = info.TargetAstFile
		logger.Debug("Definition is in the current file")
	} else {
		if pkg != nil && pkg.Syntax != nil {
			found := false
			for _, syntaxFile := range pkg.Syntax {
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
		addAnalysisError(info, fmt.Errorf("could not find AST for definition file '%s' of object '%s'", defFileName, obj.Name()), logger) // Pass logger
		return
	}

	defPath, _ := astutil.PathEnclosingInterval(defAST, defPos, defPos)
	if len(defPath) == 0 {
		addAnalysisError(info, fmt.Errorf("could not find AST path for definition of '%s' at pos %s", obj.Name(), defPosStr), logger) // Pass logger
		return
	}

	for _, node := range defPath {
		isDeclNode := false
		switch n := node.(type) {
		case *ast.FuncDecl:
			if n.Name != nil && n.Name.Pos() == defPos {
				isDeclNode = true
			}
		case *ast.ValueSpec:
			for _, name := range n.Names {
				if name != nil && name.Pos() == defPos {
					isDeclNode = true
					break
				}
			}
		case *ast.TypeSpec:
			if n.Name != nil && n.Name.Pos() == defPos {
				isDeclNode = true
			}
		case *ast.Field:
			for _, name := range n.Names {
				if name != nil && name.Pos() == defPos {
					isDeclNode = true
					break
				}
			}
		case *ast.AssignStmt:
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
			return
		}
	}

	info.IdentifierDefNode = defPath[0]
	logger.Warn("Could not pinpoint specific defining declaration node, using innermost node at definition position", "object", obj.Name(), "innermost_type", fmt.Sprintf("%T", defPath[0]))
}

// calculateArgIndex determines the 0-based index of the argument the cursor is in.
func calculateArgIndex(args []ast.Expr, cursorPos token.Pos) int {
	if len(args) == 0 {
		return 0
	}
	for i, arg := range args {
		if arg == nil {
			if i+1 < len(args) && args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i
			}
			continue
		}
		argStart := arg.Pos()
		argEnd := arg.End()
		slotStart := argStart
		if i > 0 && args[i-1] != nil {
			slotStart = args[i-1].End() + 1
		} else if i > 0 {
			slotStart = argStart
		}

		if cursorPos >= slotStart && cursorPos <= argEnd {
			return i
		}
		if cursorPos > argEnd {
			if i == len(args)-1 {
				return i + 1
			}
			if args[i+1] != nil && cursorPos < args[i+1].Pos() {
				return i + 1
			}
		}
	}
	if len(args) > 0 && args[0] != nil && cursorPos < args[0].Pos() {
		return 0
	}
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
		if argIndex >= numParams-1 {
			lastParam := params.At(numParams - 1)
			if lastParam == nil {
				return nil
			}
			if slice, ok := lastParam.Type().(*types.Slice); ok {
				return slice.Elem()
			}
			return nil
		}
		param := params.At(argIndex)
		if param == nil {
			return nil
		}
		return param.Type()
	}

	if argIndex < numParams {
		param := params.At(argIndex)
		if param == nil {
			return nil
		}
		return param.Type()
	}
	return nil
}

// listTypeMembers attempts to list exported fields and methods for a given type.
func listTypeMembers(typ types.Type, expr ast.Expr, qualifier types.Qualifier, logger *slog.Logger) []MemberInfo {
	if logger == nil {
		logger = slog.Default()
	}
	if typ == nil {
		logger.Debug("Cannot list members: input type is nil")
		return nil
	}
	logger = logger.With("type", typ.String())

	var members []MemberInfo
	seenMembers := make(map[string]MemberKind)

	msets := []*types.MethodSet{types.NewMethodSet(typ)}
	if _, isInterface := typ.Underlying().(*types.Interface); !isInterface {
		if ptrType := types.NewPointer(typ); ptrType != nil {
			msets = append(msets, types.NewMethodSet(ptrType))
		}
	}

	for _, mset := range msets {
		if mset == nil {
			continue
		}
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

	currentType := typ
	if ptr, ok := typ.(*types.Pointer); ok {
		if ptr.Elem() == nil {
			logger.Debug("Cannot list fields: pointer element type is nil")
			return members
		}
		currentType = ptr.Elem()
	}
	underlying := currentType.Underlying()
	if underlying == nil {
		logger.Debug("Cannot list members: underlying type is nil")
		return members
	}

	if st, ok := underlying.(*types.Struct); ok {
		logger.Debug("Type is a struct, listing fields", "num_fields", st.NumFields())
		for i := 0; i < st.NumFields(); i++ {
			field := st.Field(i)
			if field != nil && field.Exported() {
				fieldName := field.Name()
				if _, exists := seenMembers[fieldName]; !exists {
					members = append(members, MemberInfo{Name: fieldName, Kind: FieldMember, TypeString: types.TypeString(field.Type(), qualifier)})
					seenMembers[fieldName] = FieldMember
					logger.Debug("Added field", "name", fieldName)
				}
			}
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
