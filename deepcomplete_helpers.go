// deepcomplete_helpers.go
// Contains internal helper functions moved from deepcomplete.go
// as part of the analysis modularization (Cycle 8).
package deepcomplete

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"log/slog"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

// ============================================================================
// Internal Analysis & Preamble Helpers
// Moved here from deepcomplete.go as part of Cycle 8 refactoring.
// ============================================================================

// buildPreamble constructs the context string sent to the LLM from analysis info.
func buildPreamble(info *AstContextInfo, qualifier types.Qualifier, logger *slog.Logger) string {
	var preamble strings.Builder
	const internalPreambleLimit = 8192 // Keep internal limit higher than config limit
	currentLen := 0
	addToPreamble := func(s string) bool {
		if currentLen+len(s) < internalPreambleLimit {
			preamble.WriteString(s)
			currentLen += len(s)
			return true
		}
		return false
	}
	addTruncMarker := func(section string) {
		msg := fmt.Sprintf("//   ... (%s truncated)\n", section)
		if currentLen+len(msg) < internalPreambleLimit {
			preamble.WriteString(msg)
			currentLen += len(msg)
		} else {
			logger.Debug("Preamble limit reached even before adding truncation marker", "section", section)
		}
	}
	if !addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), info.PackageName)) {
		return preamble.String()
	}
	if !formatImportsSection(&preamble, info, addToPreamble, addTruncMarker, logger) { // Pass logger
		return preamble.String()
	}
	if !formatEnclosingFuncSection(&preamble, info, qualifier, addToPreamble) {
		return preamble.String()
	}
	if !formatCommentsSection(&preamble, info, addToPreamble, addTruncMarker, logger) { // Pass logger
		return preamble.String()
	}
	if !formatCursorContextSection(&preamble, info, qualifier, addToPreamble, logger) { // Pass logger
		return preamble.String()
	}
	formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker, logger) // Pass logger
	return preamble.String()
}

// formatImportsSection formats the import list, respecting limits.
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) bool {
	if len(info.Imports) == 0 {
		return true
	}
	if !add("// Imports:\n") {
		return false
	}
	count := 0
	maxImports := 20
	for _, imp := range info.Imports {
		if imp == nil || imp.Path == nil {
			continue
		}
		if count >= maxImports {
			addTrunc("imports")
			logger.Debug("Truncated imports list in preamble", "max_imports", maxImports)
			return true
		}
		path := imp.Path.Value
		name := ""
		if imp.Name != nil {
			name = imp.Name.Name + " "
		}
		line := fmt.Sprintf("//   import %s%s\n", name, path)
		if !add(line) {
			return false
		}
		count++
	}
	return true
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
	if info.EnclosingFunc != nil {
		name := info.EnclosingFunc.Name()
		sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		if info.ReceiverType != "" && strings.HasPrefix(sigStr, "func(") {
			sigStr = "func" + strings.TrimPrefix(sigStr, "func")
		}
		funcHeader = fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr)
	} else if info.EnclosingFuncNode != nil {
		name := "[anonymous]"
		if info.EnclosingFuncNode.Name != nil {
			name = info.EnclosingFuncNode.Name.Name
		}
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s(...)\n", funcOrMethod, receiverStr, name)
	} else {
		return true
	}
	return add(funcHeader)
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
	maxComments := 5
	for _, c := range info.CommentsNearCursor {
		if count >= maxComments {
			addTrunc("comments")
			logger.Debug("Truncated comments list in preamble", "max_comments", maxComments)
			return true
		}
		cleanComment := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(strings.TrimPrefix(c, "//")), "*/")), "/*"))
		if len(cleanComment) > 0 {
			line := fmt.Sprintf("//   %s\n", cleanComment)
			if !add(line) {
				return false
			}
			count++
		}
	}
	return true
}

// formatCursorContextSection formats specific context like calls, selectors, etc.
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, logger *slog.Logger) bool {
	// --- Function Call Context ---
	if info.CallExpr != nil {
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
		return true
	}
	// --- Selector Expression Context ---
	if info.SelectorExpr != nil {
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
		if info.SelectorExprType != nil && isKnownMember {
			members := listTypeMembers(info.SelectorExprType, info.SelectorExpr.X, qualifier)
			var fields, methods []MemberInfo
			if members != nil {
				for _, m := range members {
					if m.Kind == FieldMember {
						fields = append(fields, m)
					} else if m.Kind == MethodMember {
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
		} else if info.SelectorExprType != nil && !isKnownMember {
			if !add("//   (Cannot list members: selected member is unknown)\n") {
				return false
			}
		} else {
			if !add("//   (Cannot list members: type analysis failed for base expression)\n") {
				return false
			}
		}
		return true
	}
	// --- Composite Literal Context ---
	if info.CompositeLit != nil {
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
		return true
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
		return true
	}
	return true
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
	// Use the passed qualifier when formatting object strings.
	for name := range info.VariablesInScope {
		obj := info.VariablesInScope[name]
		items = append(items, fmt.Sprintf("//   %s\n", types.ObjectString(obj, qualifier)))
	}
	sort.Strings(items)
	count := 0
	maxScopeItems := 30
	for _, item := range items {
		if count >= maxScopeItems {
			addTrunc("scope")
			logger.Debug("Truncated scope list in preamble", "max_items", maxScopeItems)
			return true
		}
		if !add(item) {
			return false
		}
		count++
	}
	return true
}

// addAnalysisError adds a non-fatal error, avoiding duplicates based on message.
func addAnalysisError(info *AstContextInfo, err error, logger *slog.Logger) {
	if err != nil && info != nil {
		errMsg := err.Error()
		for _, existing := range info.AnalysisErrors {
			if existing.Error() == errMsg {
				return // Avoid duplicate error messages
			}
		}
		logger.Warn("Analysis warning", "error", err) // Log the warning
		info.AnalysisErrors = append(info.AnalysisErrors, err)
	}
}

// logAnalysisErrors logs joined non-fatal analysis errors.
func logAnalysisErrors(errs []error, logger *slog.Logger) {
	if len(errs) > 0 {
		combinedErr := errors.Join(errs...)
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
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		logger.Debug("No AST path found enclosing cursor position", "pos", cursorPos)
	}
	return path
}

// gatherScopeContext walks the enclosing path to find relevant scope information.
func gatherScopeContext(path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if fset == nil && path != nil {
		logger.Warn("Cannot format receiver type - fset is nil in gatherScopeContext.")
	}
	if path != nil {
		for i := len(path) - 1; i >= 0; i-- {
			node := path[i]
			switch n := node.(type) {
			case *ast.FuncDecl:
				if info.EnclosingFuncNode == nil {
					info.EnclosingFuncNode = n
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
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
					if obj, ok := targetPkg.TypesInfo.Defs[n.Name]; ok && obj != nil {
						if fn, ok := obj.(*types.Func); ok {
							if info.EnclosingFunc == nil {
								info.EnclosingFunc = fn
								if sig, ok := fn.Type().(*types.Signature); ok {
									addSignatureToScope(sig, info.VariablesInScope)
								}
							}
						}
					} else {
						if info.EnclosingFunc == nil && n.Name != nil {
							addAnalysisError(info, fmt.Errorf("definition for func '%s' not found in TypesInfo", n.Name.Name), logger)
						}
					}
				} else if info.EnclosingFuncNode != nil && info.EnclosingFunc == nil {
					reason := "type info unavailable"
					if targetPkg != nil && targetPkg.TypesInfo == nil {
						reason = "TypesInfo is nil"
					}
					if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs == nil {
						reason = "TypesInfo.Defs is nil"
					}
					funcName := "[anonymous]"
					if info.EnclosingFuncNode.Name != nil {
						funcName = info.EnclosingFuncNode.Name.Name
					}
					addAnalysisError(info, fmt.Errorf("type info for enclosing func '%s' unavailable: %s", funcName, reason), logger)
				}
			case *ast.BlockStmt:
				if info.EnclosingBlock == nil {
					info.EnclosingBlock = n
				}
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					} else {
						posStr := ""
						if fset != nil {
							posStr = fset.Position(n.Pos()).String()
						}
						addAnalysisError(info, fmt.Errorf("scope info missing for block at %s", posStr), logger)
					}
				} else if info.EnclosingBlock == n {
					reason := "type info unavailable"
					if targetPkg != nil && targetPkg.TypesInfo == nil {
						reason = "TypesInfo is nil"
					}
					if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes == nil {
						reason = "TypesInfo.Scopes is nil"
					}
					posStr := ""
					if fset != nil {
						posStr = fset.Position(n.Pos()).String()
					}
					addAnalysisError(info, fmt.Errorf("cannot get scope variables for block at %s: %s", posStr, reason), logger)
				}
			}
		}
	}
	addPackageScope(targetPkg, info, logger) // Pass logger
}

// addPackageScope adds package-level identifiers to the scope map.
func addPackageScope(targetPkg *packages.Package, info *AstContextInfo, logger *slog.Logger) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope)
		} else {
			addAnalysisError(info, fmt.Errorf("package scope missing for pkg %s", targetPkg.PkgPath), logger)
		}
	} else {
		if targetPkg != nil {
			addAnalysisError(info, fmt.Errorf("package.Types field is nil for pkg %s", targetPkg.PkgPath), logger)
		} else {
			addAnalysisError(info, errors.New("cannot add package scope: targetPkg is nil"), logger)
		}
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
				case *types.Var, *types.Const, *types.TypeName, *types.Func, *types.Label, *types.PkgName, *types.Builtin, *types.Nil:
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
func findRelevantComments(targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if targetFileAST == nil || fset == nil {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"), logger)
		return
	}
	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset)
}

// findCommentsWithMap implements the logic to find preceding or enclosing doc comments.
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		return comments
	}
	cursorLine := fset.Position(cursorPos).Line
	foundPreceding := false
	var precedingComments []string
	for node := range cmap {
		if node == nil {
			continue
		}
		for _, cg := range cmap[node] {
			if cg == nil {
				continue
			}
			if cg.End().IsValid() && fset.Position(cg.End()).Line == cursorLine-1 {
				for _, c := range cg.List {
					if c != nil {
						precedingComments = append(precedingComments, c.Text)
					}
				}
				foundPreceding = true
				break
			}
		}
		if foundPreceding {
			break
		}
	}
	if foundPreceding {
		comments = append(comments, precedingComments...)
	} else {
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
	seen := make(map[string]struct{})
	uniqueComments := make([]string, 0, len(comments))
	for _, c := range comments {
		if _, ok := seen[c]; !ok {
			seen[c] = struct{}{}
			uniqueComments = append(uniqueComments, c)
		}
	}
	return uniqueComments
}

// findContextNodes identifies specific AST nodes (call, selector, literal, ident) at/near the cursor.
func findContextNodes(path []ast.Node, cursorPos token.Pos, pkg *packages.Package, fset *token.FileSet, info *AstContextInfo, logger *slog.Logger) {
	if len(path) == 0 || fset == nil {
		if fset == nil {
			addAnalysisError(info, errors.New("Fset is nil in findContextNodes"), logger)
		}
		return
	}
	posStr := func(p token.Pos) string {
		if p.IsValid() {
			return fset.Position(p).String()
		}
		return fmt.Sprintf("Pos(%d)", p)
	}
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
	} else if pkg == nil {
		addAnalysisError(info, errors.New("cannot perform type analysis: target package is nil"), logger)
	} else {
		addAnalysisError(info, errors.New("cannot perform type analysis: pkg.TypesInfo is nil"), logger)
	}
	innermostNode := path[0]
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
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
		} else if hasTypeInfo {
			addAnalysisError(info, errors.New("cannot analyze composite literal: Types map is nil"), logger)
		}
		return
	}
	callExpr, callExprOk := path[0].(*ast.CallExpr)
	if !callExprOk && len(path) > 1 {
		callExpr, callExprOk = path[1].(*ast.CallExpr)
	}
	if callExprOk && cursorPos > callExpr.Lparen && cursorPos <= callExpr.Rparen {
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
				if ok && tv.Type == nil {
					addAnalysisError(info, fmt.Errorf("type info resolved to nil for call func (%T) at %s", callExpr.Fun, posStr(callExpr.Fun.Pos())), logger)
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info for call func (%T) at %s", callExpr.Fun, posStr(callExpr.Fun.Pos())), logger)
				}
			}
		} else if hasTypeInfo {
			addAnalysisError(info, errors.New("cannot analyze call expr: Types map is nil"), logger)
		}
		return
	}
	for i := 0; i < len(path) && i < 2; i++ {
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok && cursorPos > selExpr.X.End() {
			info.SelectorExpr = selExpr
			if hasTypeInfo && typesMap != nil {
				if tv, ok := typesMap[selExpr.X]; ok {
					info.SelectorExprType = tv.Type
					if tv.Type == nil {
						addAnalysisError(info, fmt.Errorf("missing type for selector base expr (%T) starting at %s", selExpr.X, posStr(selExpr.X.Pos())), logger)
					}
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info entry for selector base expr (%T) starting at %s", selExpr.X, posStr(selExpr.X.Pos())), logger)
				}
			} else if hasTypeInfo {
				addAnalysisError(info, errors.New("cannot analyze selector expr: Types map is nil"), logger)
			}
			if info.SelectorExprType != nil && selExpr.Sel != nil {
				selName := selExpr.Sel.Name
				obj, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
				if obj == nil {
					addAnalysisError(info, fmt.Errorf("selecting unknown member '%s' from type '%s'", selName, info.SelectorExprType.String()), logger)
				}
			}
			return
		}
	}
	var ident *ast.Ident
	if id, ok := path[0].(*ast.Ident); ok && cursorPos == id.End() {
		ident = id
	} else if len(path) > 1 {
		if id, ok := path[1].(*ast.Ident); ok && cursorPos >= id.Pos() && cursorPos <= id.End() {
			if _, pIsSel := path[0].(*ast.SelectorExpr); !pIsSel || path[0].(*ast.SelectorExpr).Sel != id {
				ident = id
			}
		}
	}
	if ident != nil {
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
			} else {
				if typesMap != nil {
					if tv, ok := typesMap[ident]; ok && tv.Type != nil {
						info.IdentifierType = tv.Type
					} else {
						if defsMap != nil && usesMap != nil {
							addAnalysisError(info, fmt.Errorf("missing object and type info for identifier '%s' at %s", ident.Name, posStr(ident.Pos())), logger)
						}
					}
				} else if defsMap == nil && usesMap == nil {
					addAnalysisError(info, fmt.Errorf("missing object info for identifier '%s' at %s (defs/uses/types maps nil)", ident.Name, posStr(ident.Pos())), logger)
				} else {
					addAnalysisError(info, fmt.Errorf("object not found for identifier '%s' at %s", ident.Name, posStr(ident.Pos())), logger)
				}
			}
		} else {
			addAnalysisError(info, errors.New("missing type info for identifier analysis"), logger)
		}
	}
}

// calculateArgIndex determines the 0-based index of the argument the cursor is in.
func calculateArgIndex(args []ast.Expr, cursorPos token.Pos) int {
	if len(args) == 0 {
		return 0
	}
	for i, arg := range args {
		if arg == nil {
			continue
		}
		argStart := arg.Pos()
		argEnd := arg.End()
		slotStart := argStart
		if i > 0 && args[i-1] != nil {
			slotStart = args[i-1].End() + 1
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
	return 0
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
	} else {
		if argIndex < numParams {
			param := params.At(argIndex)
			if param == nil {
				return nil
			}
			return param.Type()
		}
	}
	return nil
}

// listTypeMembers attempts to list exported fields and methods for a given type.
func listTypeMembers(typ types.Type, expr ast.Expr, qualifier types.Qualifier) []MemberInfo {
	if typ == nil {
		return nil
	}
	var members []MemberInfo
	currentType := typ
	isPointer := false
	if ptr, ok := typ.(*types.Pointer); ok {
		if ptr.Elem() == nil {
			return nil
		}
		currentType = ptr.Elem()
		isPointer = true
	}
	if currentType == nil {
		return nil
	}
	underlying := currentType.Underlying()
	if underlying == nil {
		return nil
	}
	switch u := underlying.(type) {
	case *types.Struct:
		members = append(members, listStructFields(u, qualifier)...)
	case *types.Interface:
		for i := 0; i < u.NumExplicitMethods(); i++ {
			method := u.ExplicitMethod(i)
			if method != nil && method.Exported() {
				members = append(members, MemberInfo{Name: method.Name(), Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
			}
		}
		for i := 0; i < u.NumEmbeddeds(); i++ {
			embeddedType := u.EmbeddedType(i)
			if embeddedType != nil {
				members = append(members, MemberInfo{Name: "// embeds", Kind: OtherMember, TypeString: types.TypeString(embeddedType, qualifier)})
			}
		}
	}
	mset := types.NewMethodSet(typ)
	for i := 0; i < mset.Len(); i++ {
		sel := mset.At(i)
		if sel != nil {
			methodObj := sel.Obj()
			if method, ok := methodObj.(*types.Func); ok {
				if method != nil && method.Exported() {
					members = append(members, MemberInfo{Name: method.Name(), Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
				}
			}
		}
	}
	if !isPointer {
		var ptrType types.Type
		if _, isNamed := currentType.(*types.Named); isNamed {
			ptrType = types.NewPointer(currentType)
		} else if _, isBasic := currentType.(*types.Basic); isBasic {
			ptrType = types.NewPointer(currentType)
		}
		if ptrType != nil {
			msetPtr := types.NewMethodSet(ptrType)
			existingMethods := make(map[string]struct{})
			for _, m := range members {
				if m.Kind == MethodMember {
					existingMethods[m.Name] = struct{}{}
				}
			}
			for i := 0; i < msetPtr.Len(); i++ {
				sel := msetPtr.At(i)
				if sel != nil {
					methodObj := sel.Obj()
					if method, ok := methodObj.(*types.Func); ok {
						if method != nil && method.Exported() {
							if _, exists := existingMethods[method.Name()]; !exists {
								members = append(members, MemberInfo{Name: method.Name(), Kind: MethodMember, TypeString: types.TypeString(method.Type(), qualifier)})
							}
						}
					}
				}
			}
		}
	}
	if len(members) > 0 {
		seen := make(map[string]struct{})
		uniqueMembers := make([]MemberInfo, 0, len(members))
		for _, m := range members {
			key := string(m.Kind) + ":" + m.Name
			if _, ok := seen[key]; !ok {
				seen[key] = struct{}{}
				uniqueMembers = append(uniqueMembers, m)
			}
		}
		members = uniqueMembers
	}
	return members
}

// listStructFields lists exported fields of a struct type.
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
