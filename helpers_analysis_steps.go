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
		scopeInfo, scopeErr := analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, line, col)
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information when targetFile is nil", "error", scopeErr)
			addAnalysisError(info, scopeErr, logger)
		} else if scopeInfo != nil {
			info.VariablesInScope = scopeInfo.Variables
		}
		return nil
	}

	cursorPos, posErr := calculateCursorPos(targetFile, line, col, logger)
	if posErr != nil {
		err := fmt.Errorf("cannot calculate valid cursor position: %w", posErr)
		addAnalysisError(info, err, logger)
		scopeInfo, scopeErr := analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, line, col)
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information after cursor error", "error", scopeErr)
			addAnalysisError(info, scopeErr, logger)
		} else if scopeInfo != nil {
			info.VariablesInScope = scopeInfo.Variables
		}
		return err
	}
	info.CursorPos = cursorPos
	logger = logger.With("cursorPos", info.CursorPos, "cursorPosStr", fset.PositionFor(info.CursorPos, true).String())
	logger.Debug("Calculated cursor position for analysis steps")

	if targetFileAST != nil {
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

		scopeInfo, scopeErr := analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, line, col)
		if scopeErr != nil {
			logger.Warn("Error getting scope info during analysis steps", "error", scopeErr)
			addAnalysisError(info, scopeErr, logger)
		} else if scopeInfo != nil {
			info.VariablesInScope = scopeInfo.Variables
		}

		comments, commentErr := analyzer.GetRelevantComments(ctx, info.FilePath, info.Version, line, col)
		if commentErr != nil {
			logger.Warn("Error getting relevant comments during analysis steps", "error", commentErr)
			addAnalysisError(info, commentErr, logger)
		}
		info.CommentsNearCursor = comments

		path, pathErr := findEnclosingPath(ctx, targetFileAST, cursorPos, info, logger)
		if pathErr != nil {
			addAnalysisError(info, fmt.Errorf("finding enclosing AST path failed in analysis steps: %w", pathErr), logger)
		}
		findContextNodes(ctx, path, cursorPos, targetPkg, fset, analyzer, info, logger)
		addAnalysisDiagnostics(fset, info, logger)

	} else {
		addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil"), logger)
		scopeInfo, scopeErr := analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, line, col)
		if scopeErr != nil {
			logger.Warn("Error extracting package scope information when targetFileAST is nil", "error", scopeErr)
			addAnalysisError(info, scopeErr, logger)
		} else if scopeInfo != nil {
			info.VariablesInScope = scopeInfo.Variables
		}
	}
	return nil
}

// findEnclosingPath finds the AST node path from the root to the node enclosing the cursor.
func findEnclosingPath(ctx context.Context, targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo, logger *slog.Logger) ([]ast.Node, error) {
	if targetFileAST == nil {
		err := errors.New("cannot find enclosing path: targetFileAST is nil")
		return nil, err
	}
	if !cursorPos.IsValid() {
		err := errors.New("cannot find enclosing path: invalid cursor position")
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		logger.Debug("No AST path found enclosing cursor position", "pos", cursorPos)
	}
	return path, nil
}

// gatherScopeContext walks the enclosing AST path to find relevant scope information
// (package, block, function parameters/receivers/named results).
// It populates info.VariablesInScope, info.EnclosingFunc, info.EnclosingFuncNode,
// info.ReceiverType, and info.EnclosingBlock.
// Non-fatal errors encountered are added to info.AnalysisErrors.
func gatherScopeContext(ctx context.Context, path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if info.VariablesInScope == nil {
		info.VariablesInScope = make(map[string]types.Object)
	}

	addPackageScope(ctx, targetPkg, info, logger)

	if path != nil {
		for i := len(path) - 1; i >= 0; i-- {
			node := path[i]
			switch n := node.(type) {
			case *ast.File:
				continue

			case *ast.FuncDecl:
				if info.EnclosingFuncNode == nil {
					info.EnclosingFuncNode = n
					if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
						if obj, ok := targetPkg.TypesInfo.Defs[n.Name].(*types.Func); ok {
							info.EnclosingFunc = obj
							if sig, okSig := obj.Type().(*types.Signature); okSig && sig.Recv() != nil {
								info.ReceiverType = types.TypeString(sig.Recv().Type(), types.RelativeTo(targetPkg.Types))
							}
						}
					}
				}
				if n.Recv != nil && len(n.Recv.List) > 0 {
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
				if n.Type != nil && n.Type.Params != nil {
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
				if n.Type != nil && n.Type.Results != nil {
					for _, field := range n.Type.Results.List {
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
				if info.EnclosingBlock == nil {
					info.EnclosingBlock = n
				}
				blockPosStr := getPosString(fset, n.Pos())
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					} else {
						addAnalysisError(info, fmt.Errorf("scope info missing for block at %s", blockPosStr), logger)
					}
				} else if info.EnclosingBlock == n {
					reason := getMissingTypeInfoReason(targetPkg)
					if !strings.Contains(errors.Join(info.AnalysisErrors...).Error(), "type info") {
						addAnalysisError(info, fmt.Errorf("cannot get scope variables for block at %s: %s", blockPosStr, reason), logger)
					}
				}
			}
		}
		if info.EnclosingFunc != nil {
			if sig, ok := info.EnclosingFunc.Type().(*types.Signature); ok {
				addSignatureToScope(sig, info.VariablesInScope)
			}
		}
	} else {
		logger.Debug("AST path is nil, cannot gather block/function specific scopes. Only package scope will be available.")
	}
}

// addPackageScope adds package-level identifiers to the scope map.
func addPackageScope(ctx context.Context, targetPkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope)
		} else {
			addAnalysisError(info, fmt.Errorf("package scope (targetPkg.Types.Scope()) missing for pkg %s", targetPkg.PkgPath), logger)
		}
	} else {
		reason := "targetPkg is nil"
		if targetPkg != nil {
			reason = fmt.Sprintf("package.Types field is nil for pkg %s", targetPkg.PkgPath)
		}
		addAnalysisError(info, fmt.Errorf("cannot add package scope: %s", reason), logger)
	}
}

// addScopeVariables adds identifiers from a types.Scope to scopeMap if they are declared before cursorPos.
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

// addSignatureToScope adds named parameters and results from a types.Signature to the scope map.
func addSignatureToScope(sig *types.Signature, scopeMap map[string]types.Object) {
	if sig == nil {
		return
	}
	addTupleToScope(sig.Params(), scopeMap)
	addTupleToScope(sig.Results(), scopeMap)
}

// addTupleToScope adds named variables from a types.Tuple (like params or results) to the scope map.
func addTupleToScope(tuple *types.Tuple, scopeMap map[string]types.Object) {
	if tuple == nil {
		return
	}
	for j := 0; j < tuple.Len(); j++ {
		v := tuple.At(j)
		if v != nil && v.Name() != "" && v.Name() != "_" {
			if _, exists := scopeMap[v.Name()]; !exists {
				scopeMap[v.Name()] = v
			}
		}
	}
}

// findRelevantComments uses ast.CommentMap to find comments near the cursor.
// It populates info.CommentsNearCursor.
func findRelevantComments(ctx context.Context, targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if targetFileAST == nil || fset == nil {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"), logger)
		if info.CommentsNearCursor == nil {
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

	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	if cmap == nil {
		addAnalysisError(info, errors.New("failed to create ast.CommentMap"), logger)
		if info.CommentsNearCursor == nil {
			info.CommentsNearCursor = []string{}
		}
		return
	}
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset, logger)
}

// findCommentsWithMap implements the logic to find preceding or enclosing doc comments using CommentMap.
// It returns a slice of comment strings.
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, logger *slog.Logger) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		logger.Debug("Skipping comment finding: cmap, cursorPos, or fset invalid/nil.")
		return comments
	}

	cursorPosInfo := fset.Position(cursorPos)
	cursorLine := cursorPosInfo.Line
	var precedingComments []string
	foundPrecedingOnLine := false
	minCommentPosOnPrecedingLine := token.Pos(-1)

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

			if commentEndLine == cursorLine-1 { // Comment ends on the line just before the cursor's line
				if !foundPrecedingOnLine || cg.Pos() > minCommentPosOnPrecedingLine {
					precedingComments = nil // Take the last comment block on that line
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
		logger.Debug("No comments on preceding line, searching for doc comments on enclosing AST nodes.")
		if path != nil { // Search up the AST path for .Doc fields
			for i := 0; i < len(path); i++ { // path[0] is innermost
				node := path[i]
				var docComment *ast.CommentGroup
				switch n := node.(type) {
				case *ast.FuncDecl:
					docComment = n.Doc
				case *ast.GenDecl:
					docComment = n.Doc // For var, const, type blocks
				case *ast.TypeSpec:
					docComment = n.Doc
				case *ast.ValueSpec:
					docComment = n.Doc // For individual var/const in a GenDecl
				case *ast.Field:
					docComment = n.Doc // Struct fields, interface methods
				}
				if docComment != nil {
					logger.Debug("Found doc comment on an enclosing AST node", "node_type", fmt.Sprintf("%T", node))
					for _, c := range docComment.List {
						if c != nil {
							comments = append(comments, c.Text)
						}
					}
					goto cleanupAndUnique // Found a doc comment, use it and stop.
				}
			}
		}
	}

cleanupAndUnique:
	if len(comments) > 1 { // Remove duplicates if any
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
func findContextNodes(
	ctx context.Context,
	path []ast.Node,
	cursorPos token.Pos,
	pkg *packages.Package,
	fset *token.FileSet,
	analyzer Analyzer,
	info *AstContextInfo,
	logger *slog.Logger,
) {
	if len(path) == 0 {
		logger.Debug("Cannot find context nodes: AST path is empty.")
		return
	}
	if fset == nil {
		addAnalysisError(info, errors.New("Fset is nil in findContextNodes, cannot proceed"), logger)
		return
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
			addAnalysisError(info, errors.New("type info map 'Types' (pkg.TypesInfo.Types) is nil"), logger)
		}
		if defsMap == nil {
			addAnalysisError(info, errors.New("type info map 'Defs' (pkg.TypesInfo.Defs) is nil"), logger)
		}
		if usesMap == nil {
			addAnalysisError(info, errors.New("type info map 'Uses' (pkg.TypesInfo.Uses) is nil"), logger)
		}
	} else {
		reason := getMissingTypeInfoReason(pkg)
		addAnalysisError(info, fmt.Errorf("cannot perform full type analysis for context nodes: %s", reason), logger)
	}

	innermostNode := path[0]
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
		compLitPosStr := posStr(compLit.Pos())
		logger.Debug("Cursor is inside a Composite Literal", "pos", compLitPosStr)
		info.CompositeLit = compLit
		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[compLit]; ok {
				info.CompositeLitType = tv.Type
				if info.CompositeLitType == nil {
					addAnalysisError(info, fmt.Errorf("composite literal type resolved to nil at %s", compLitPosStr), logger)
				}
			} else {
				addAnalysisError(info, fmt.Errorf("missing type info for composite literal at %s", compLitPosStr), logger)
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
		logger.Debug("Cursor is inside Call Expression arguments", "call_pos", callPosStr, "func_pos", funPosStr)
		info.CallExpr = callExpr
		info.CallArgIndex = calculateArgIndex(callExpr.Args, cursorPos)

		if hasTypeInfo && typesMap != nil {
			if tv, ok := typesMap[callExpr.Fun]; ok && tv.Type != nil {
				if sig, okSig := tv.Type.Underlying().(*types.Signature); okSig {
					info.CallExprFuncType = sig
					info.ExpectedArgType = determineExpectedArgType(sig, info.CallArgIndex)
				} else {
					addAnalysisError(info, fmt.Errorf("type of call func (%T) at %s is %T, not a signature", callExpr.Fun, funPosStr, tv.Type), logger)
				}
			} else {
				reason := "missing type info entry"
				if tv, ok := typesMap[callExpr.Fun]; ok && tv.Type == nil {
					reason = "type info resolved to nil"
				}
				addAnalysisError(info, fmt.Errorf("%s for call func (%T) at %s", reason, callExpr.Fun, funPosStr), logger)
			}
		}
		return
	}

	for i := 0; i < len(path) && i < 2; i++ {
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok && cursorPos > selExpr.X.End() {
			selPosStr := posStr(selExpr.Pos())
			basePosStr := posStr(selExpr.X.Pos())
			logger.Debug("Cursor is inside a Selector Expression (after '.')", "selector_pos", selPosStr, "base_expr_pos", basePosStr)
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
			return
		}
	}

	var ident *ast.Ident
	if id, ok := path[0].(*ast.Ident); ok && cursorPos >= id.Pos() && cursorPos <= id.End() {
		isSelectorSel, isCallFun := false, false
		if len(path) > 1 {
			if sel, okSel := path[1].(*ast.SelectorExpr); okSel && sel.Sel == id {
				isSelectorSel = true
			}
			if call, okCall := path[1].(*ast.CallExpr); okCall && call.Fun == id {
				isCallFun = true
			} else if len(path) > 2 {
				if selAsFun, okSelFun := path[1].(*ast.SelectorExpr); okSelFun {
					if call, okCallParent := path[2].(*ast.CallExpr); okCallParent && call.Fun == selAsFun && selAsFun.Sel == id {
						isCallFun = true
					}
				}
			}
		}
		if !isSelectorSel && !isCallFun {
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
			if usesMap != nil {
				obj = usesMap[ident]
			}
			if obj == nil && defsMap != nil {
				obj = defsMap[ident]
			}

			if obj != nil {
				info.IdentifierObject = obj
				info.IdentifierType = obj.Type()
				defObjPosStr := posStr(obj.Pos())
				if info.IdentifierType == nil {
					addAnalysisError(info, fmt.Errorf("object '%s' at %s found but its type is nil", obj.Name(), defObjPosStr), logger)
				}
				findDefiningNode(ctx, obj, fset, pkg, info, logger)
			} else {
				if typesMap != nil {
					if tv, ok := typesMap[ident]; ok && tv.Type != nil {
						info.IdentifierType = tv.Type
						logger.Debug("Identifier object not in Uses/Defs, but type found in Types map.", "name", ident.Name, "type", tv.Type.String())
					} else {
						addAnalysisError(info, fmt.Errorf("missing object and type info for identifier '%s' at %s", ident.Name, identPosStr), logger)
					}
				} else {
					addAnalysisError(info, fmt.Errorf("object not found for identifier '%s' at %s (and Types map is nil)", ident.Name, identPosStr), logger)
				}
			}
		} else {
			addAnalysisError(info, errors.New("missing general type information for identifier analysis"), logger)
		}
	}
	logger.Debug("Finished context node identification.")
}

// findDefiningNode attempts to find the AST node where a types.Object is defined.
// This is crucial for features like "go to definition" and providing documentation on hover.
// It populates info.IdentifierDefNode.
func findDefiningNode(ctx context.Context, obj types.Object, fset *token.FileSet, pkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	defPos := obj.Pos()
	if !defPos.IsValid() {
		logger.Debug("Object has an invalid definition position, cannot find defining AST node.", "object_name", obj.Name())
		return
	}

	defPosStr := getPosString(fset, defPos)
	defFileToken := fset.File(defPos)
	if defFileToken == nil {
		addAnalysisError(info, fmt.Errorf("could not find token.File for definition of '%s' at pos %s", obj.Name(), defPosStr), logger)
		return
	}
	defFileName := defFileToken.Name()
	logger = logger.With("def_file", defFileName)

	var defAST *ast.File
	if defFileName == info.FilePath {
		defAST = info.TargetAstFile
		logger.Debug("Definition of object is in the current file.", "object_name", obj.Name())
	} else {
		if pkg != nil && pkg.Syntax != nil {
			found := false
			for _, syntaxFile := range pkg.Syntax {
				if syntaxFile != nil && fset.File(syntaxFile.Pos()) == defFileToken {
					defAST = syntaxFile
					found = true
					logger.Debug("Found definition AST in another file within the package's syntax trees.", "object_name", obj.Name())
					break
				}
			}
			if !found {
				logger.Warn("Definition file AST not found within the loaded package's syntax trees.", "package_id", pkg.ID, "object_name", obj.Name())
			}
		} else {
			reason := "package is nil"
			if pkg != nil {
				reason = "package.Syntax is nil"
			}
			logger.Warn("Cannot search for definition AST in other files.", "reason", reason, "object_name", obj.Name())
		}
	}

	if defAST == nil {
		addAnalysisError(info, fmt.Errorf("could not find or load AST for definition file '%s' of object '%s'", defFileName, obj.Name()), logger)
		return
	}

	defPath, _ := astutil.PathEnclosingInterval(defAST, defPos, defPos)
	if len(defPath) == 0 {
		addAnalysisError(info, fmt.Errorf("could not find AST path for definition of '%s' at pos %s in file %s", obj.Name(), defPosStr, defFileName), logger)
		return
	}

	// Corrected logic: Use a switch statement for type assertion.
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
			// Handle unnamed fields like embedded types or interface methods without explicit names at this Pos.
			if len(n.Names) == 0 && n.Type != nil && n.Type.Pos() == defPos {
				isDeclNode = true
			}
		case *ast.AssignStmt:
			if n.Tok == token.DEFINE { // Check for short variable declaration `:=`
				for _, lhsExpr := range n.Lhs {
					if id, ok := lhsExpr.(*ast.Ident); ok && id.Pos() == defPos {
						isDeclNode = true
						break
					}
				}
			}
		case *ast.ImportSpec:
			// For import "path" or name "path"
			if n.Path != nil && n.Path.Pos() == defPos { // Check if object's Pos matches the import path's Pos
				isDeclNode = true
			} else if n.Name != nil && n.Name.Pos() == defPos { // Check if object's Pos matches the named import's Pos
				isDeclNode = true
			}
		}

		if isDeclNode {
			info.IdentifierDefNode = node
			logger.Debug("Found defining AST node for object.", "object_name", obj.Name(), "node_type", fmt.Sprintf("%T", node), "node_pos", getPosString(fset, node.Pos()))
			return
		}
	}

	// Fallback: if no specific declaration node matched exactly,
	// use the innermost node at the definition position.
	info.IdentifierDefNode = defPath[0]
	logger.Warn("Could not pinpoint specific defining declaration node, using innermost AST node at definition position as fallback.", "object_name", obj.Name(), "innermost_node_type", fmt.Sprintf("%T", defPath[0]))
}

// calculateArgIndex determines the 0-based index of the argument the cursor is in within a call expression.
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
		argStart, argEnd := arg.Pos(), arg.End()
		slotStart := argStart
		if i > 0 {
			prevArg := args[i-1]
			if prevArg != nil {
				slotStart = prevArg.End() + 1
			}
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

// determineExpectedArgType finds the expected type for a given argument index in a function signature.
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
	} else {
		if argIndex < numParams {
			param := params.At(argIndex)
			if param == nil {
				return nil
			}
			return param.Type()
		}
		return nil
	}
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
	logger = logger.With("type_string", typ.String())

	var members []MemberInfo
	seenMembers := make(map[string]MemberKind)

	msets := []*types.MethodSet{types.NewMethodSet(typ)}
	if _, isInterface := typ.Underlying().(*types.Interface); !isInterface {
		if _, isPointer := typ.(*types.Pointer); !isPointer {
			if ptrType := types.NewPointer(typ); ptrType != nil {
				msets = append(msets, types.NewMethodSet(ptrType))
			}
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
					members = append(members, MemberInfo{
						Name: methodName, Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier),
					})
					seenMembers[methodName] = MethodMember
					logger.Debug("Added method to member list", "method_name", methodName)
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
		currentType = ptr.Elem().Underlying()
	} else {
		currentType = typ.Underlying()
	}

	if currentType == nil {
		logger.Debug("Cannot list fields: underlying type is nil after potential dereference.")
		return members
	}

	if st, ok := currentType.(*types.Struct); ok {
		logger.Debug("Type is a struct, listing its fields.", "num_fields_in_struct", st.NumFields())
		for i := 0; i < st.NumFields(); i++ {
			field := st.Field(i)
			if field != nil && field.Exported() {
				fieldName := field.Name()
				if _, exists := seenMembers[fieldName]; !exists {
					members = append(members, MemberInfo{
						Name: fieldName, Kind: FieldMember, TypeString: types.TypeString(field.Type(), qualifier),
					})
					seenMembers[fieldName] = FieldMember
					logger.Debug("Added field to member list", "field_name", fieldName)
				}
			}
		}
	} else if iface, ok := currentType.(*types.Interface); ok {
		logger.Debug("Type is an interface. Explicit methods already covered by MethodSet.", "num_explicit_methods", iface.NumExplicitMethods())
	} else {
		logger.Debug("Type is not a struct or interface, no direct fields to list.", "actual_type_kind", fmt.Sprintf("%T", currentType))
	}

	logger.Debug("Finished listing type members", "total_members_found", len(members))
	return members
}
