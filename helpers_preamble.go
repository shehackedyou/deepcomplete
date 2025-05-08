// deepcomplete/helpers_preamble.go
// Contains helper functions specifically for constructing the LLM prompt preamble.
package deepcomplete

import (
	"fmt"
	"go/ast"
	"go/types"
	"log/slog"
	"path/filepath"
	"sort"
	"strings"
)

// ============================================================================
// Preamble Construction Helpers
// ============================================================================

// constructPromptPreamble builds the final preamble string from analyzed info.
func constructPromptPreamble(
	analyzer Analyzer, // Use interface type
	info *AstContextInfo,
	qualifier types.Qualifier,
	logger *slog.Logger,
) string {
	return buildPreamble(analyzer, info, qualifier, logger)
}

// buildPreamble constructs the context string sent to the LLM from analysis info.
func buildPreamble(
	analyzer Analyzer, // Use interface type
	info *AstContextInfo,
	qualifier types.Qualifier,
	logger *slog.Logger,
) string {
	if logger == nil {
		logger = slog.Default()
	}
	var preamble strings.Builder
	const internalPreambleLimit = 8192 // Limit overall preamble construction size
	currentLen := 0
	limitReached := false

	// Helper to add string to preamble if within limit
	addToPreamble := func(s string) bool {
		if limitReached {
			return false
		}
		if currentLen+len(s) < internalPreambleLimit {
			preamble.WriteString(s)
			currentLen += len(s)
			return true
		}
		limitReached = true
		logger.Debug("Internal preamble construction limit reached", "limit", internalPreambleLimit, "current_len", currentLen)
		return false
	}

	// Helper to add truncation marker if limit wasn't already hit
	addTruncMarker := func(section string) {
		if limitReached {
			return
		}
		msg := fmt.Sprintf("// ... (%s truncated)\n", section)
		if currentLen+len(msg) < internalPreambleLimit {
			preamble.WriteString(msg)
			currentLen += len(msg)
		}
		limitReached = true // Mark limit reached even if marker couldn't fit
	}

	// --- Build Preamble Sections ---
	pkgName := info.PackageName
	if pkgName == "" && info.TargetPackage != nil && info.TargetPackage.Types != nil {
		pkgName = info.TargetPackage.Types.Name()
	}
	if pkgName == "" {
		pkgName = "[unknown]"
	}
	addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), pkgName))

	// Add sections sequentially, checking the limit after each potentially large section
	if !limitReached {
		formatImportsSection(&preamble, info, addToPreamble, addTruncMarker, logger)
	}
	if !limitReached {
		formatEnclosingFuncSection(&preamble, info, qualifier, addToPreamble, logger)
	}
	if !limitReached {
		formatCommentsSection(&preamble, info, addToPreamble, addTruncMarker, logger)
	}
	if !limitReached {
		formatCursorContextSection(&preamble, info, qualifier, addToPreamble, logger)
	}
	if !limitReached {
		formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker, logger)
	}

	return preamble.String()
}

// formatImportsSection formats the import list, respecting limits.
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if len(info.Imports) == 0 {
		return
	}
	if !add("// Imports:\n") {
		return
	}

	count := 0
	const maxImports = 25
	for _, imp := range info.Imports {
		if imp == nil || imp.Path == nil {
			logger.Warn("Skipping nil import spec or import path during preamble generation")
			continue
		}
		if count >= maxImports {
			addTrunc("imports")
			logger.Debug("Truncated imports list in preamble", "max_imports", maxImports)
			return
		}
		path := imp.Path.Value
		name := ""
		if imp.Name != nil {
			name = imp.Name.Name + " "
		}
		line := fmt.Sprintf("//   import %s%s\n", name, path)
		if !add(line) {
			return
		}
		count++
	}
}

// formatEnclosingFuncSection formats the enclosing function/method info.
func formatEnclosingFuncSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, logger *slog.Logger) {
	funcOrMethod := "Function"
	receiverStr := ""
	if info.ReceiverType != "" {
		funcOrMethod = "Method"
		receiverBase := info.ReceiverType
		if strings.HasPrefix(receiverBase, "*") {
			receiverBase = receiverBase[1:]
		}
		receiverStr = fmt.Sprintf("(%s) ", receiverBase)
	}

	var funcHeader string
	if info.EnclosingFunc != nil { // Prefer type information
		name := info.EnclosingFunc.Name()
		sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		if info.ReceiverType != "" && strings.HasPrefix(sigStr, "func(") {
			if firstParen := strings.Index(sigStr, "("); firstParen != -1 {
				if secondParen := strings.Index(sigStr[firstParen+1:], ")"); secondParen != -1 {
					sigStr = "func" + sigStr[firstParen+1+secondParen+1:]
				}
			}
		}
		funcHeader = fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr)
	} else if info.EnclosingFuncNode != nil { // Fallback to AST
		name := "[anonymous]"
		if info.EnclosingFuncNode.Name != nil {
			name = info.EnclosingFuncNode.Name.Name
		}
		paramsStr := "..."
		resultsStr := ""
		if info.EnclosingFuncNode.Type != nil {
			if info.EnclosingFuncNode.Type.Params != nil {
				paramsStr = fmt.Sprintf("(%d params)", len(info.EnclosingFuncNode.Type.Params.List))
			}
			if info.EnclosingFuncNode.Type.Results != nil {
				resultsStr = fmt.Sprintf(" (%d results)", len(info.EnclosingFuncNode.Type.Results.List))
			}
		}
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s%s%s\n", funcOrMethod, receiverStr, name, paramsStr, resultsStr)
	} else {
		return
	}
	add(funcHeader)
}

// formatCommentsSection formats relevant comments, respecting limits.
func formatCommentsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if len(info.CommentsNearCursor) == 0 {
		return
	}
	if !add("// Relevant Comments:\n") {
		return
	}

	count := 0
	const maxComments = 7
	for _, c := range info.CommentsNearCursor {
		if count >= maxComments {
			addTrunc("comments")
			logger.Debug("Truncated comments list in preamble", "max_comments", maxComments)
			return
		}
		cleanComment := strings.TrimSpace(strings.TrimPrefix(c, "//"))
		cleanComment = strings.TrimSpace(strings.TrimPrefix(cleanComment, "/*"))
		cleanComment = strings.TrimSpace(strings.TrimSuffix(cleanComment, "*/"))

		lines := strings.Split(cleanComment, "\n")
		for _, commentLine := range lines {
			trimmedLine := strings.TrimSpace(commentLine)
			if len(trimmedLine) > 0 {
				line := fmt.Sprintf("//   %s\n", trimmedLine)
				if !add(line) {
					return
				}
				count++
				if count >= maxComments {
					addTrunc("comments")
					logger.Debug("Truncated comments list in preamble (mid-comment)", "max_comments", maxComments)
					return
				}
			}
		}
	}
}

// formatCursorContextSection formats specific context like calls, selectors, etc.
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, logger *slog.Logger) {
	if logger == nil {
		logger = slog.Default()
	}
	contextAdded := false

	// --- Function Call Context ---
	if info.CallExpr != nil {
		contextAdded = true
		funcName := "[unknown function]"
		// var funcObj types.Object // Removed unused variable

		// Correctly get the identifier for lookup
		var funcIdent *ast.Ident
		switch fun := info.CallExpr.Fun.(type) {
		case *ast.Ident:
			funcIdent = fun
			funcName = fun.Name // Use AST name as fallback
		case *ast.SelectorExpr:
			funcIdent = fun.Sel
			funcName = fun.Sel.Name // Use AST name as fallback
		}

		// Look up the object in TypesInfo to get the accurate name if possible
		if funcIdent != nil && info.TargetPackage != nil && info.TargetPackage.TypesInfo != nil {
			if obj, ok := info.TargetPackage.TypesInfo.Uses[funcIdent]; ok && obj != nil {
				// funcObj = obj // Removed unused assignment
				funcName = obj.Name() // Use name from types.Object if found
			} else if obj, ok := info.TargetPackage.TypesInfo.Defs[funcIdent]; ok && obj != nil {
				// funcObj = obj // Removed unused assignment
				funcName = obj.Name()
			}
		}

		if !add(fmt.Sprintf("// Inside function call: %s (Arg Index %d)\n", funcName, info.CallArgIndex)) {
			return
		}

		// Use the function signature stored in info (if analysis succeeded)
		if sig := info.CallExprFuncType; sig != nil {
			if !add(fmt.Sprintf("//   Signature: %s\n", types.TypeString(sig, qualifier))) {
				return
			}
			params := sig.Params()
			if params != nil && info.CallArgIndex >= 0 && info.CallArgIndex < params.Len() {
				p := params.At(info.CallArgIndex)
				if p != nil {
					paramStr := types.ObjectString(p, qualifier)
					if paramStr == "" {
						paramStr = fmt.Sprintf("[unnamed %s]", types.TypeString(p.Type(), qualifier))
					}
					isVariadic := sig.Variadic() && info.CallArgIndex == params.Len()-1
					variadicStr := ""
					if isVariadic {
						variadicStr = " (variadic)"
					}
					if !add(fmt.Sprintf("//   Current Param: %s%s\n", paramStr, variadicStr)) {
						return
					}
				}
			}
		} else {
			logger.Warn("Could not determine function signature for call expression", "function_name", funcName)
			if !add("//   Signature: (unknown)\n") {
				return
			}
		}
	}

	// --- Selector Expression Context ---
	if info.SelectorExpr != nil && !contextAdded {
		contextAdded = true
		selName := ""
		if info.SelectorExpr.Sel != nil {
			selName = info.SelectorExpr.Sel.Name
		}
		typeName := "(unknown type)"
		if info.SelectorExprType != nil {
			typeName = types.TypeString(info.SelectorExprType, qualifier)
		}

		fieldOrMethod, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
		isKnownMember := fieldOrMethod != nil
		status := ""
		if info.SelectorExprType != nil && selName != "" {
			if isKnownMember {
				status = fmt.Sprintf(" (selecting '%s')", selName)
			} else {
				status = fmt.Sprintf(" (selecting unknown member '%s')", selName)
			}
		} else if selName != "" {
			status = fmt.Sprintf(" (selecting '%s')", selName)
		}

		if !add(fmt.Sprintf("// Selector context: Base Type = %s%s\n", typeName, status)) {
			return
		}

		if info.SelectorExprType != nil && (!isKnownMember || selName == "") {
			members := listTypeMembers(info.SelectorExprType, info.SelectorExpr.X, qualifier, logger)
			if len(members) > 0 {
				if !add("//   Available Members (sample):\n") {
					return
				}
				sort.Slice(members, func(i, j int) bool { return members[i].Name < members[j].Name })
				count := 0
				const maxMembers = 5
				for _, m := range members {
					if count >= maxMembers {
						add("//     ...\n")
						break
					}
					memberSig := m.TypeString
					if m.Kind == MethodMember {
						memberSig = strings.TrimPrefix(m.TypeString, "func")
					}
					if !add(fmt.Sprintf("//     - %s [%s] %s\n", m.Name, m.Kind, memberSig)) {
						return
					}
					count++
				}
			} else {
				if !add("//   (No exported members found or type analysis failed)\n") {
					return
				}
			}
		}
	}

	// --- Composite Literal Context ---
	if info.CompositeLit != nil && !contextAdded {
		contextAdded = true
		typeName := "(unknown type)"
		if info.CompositeLitType != nil {
			typeName = types.TypeString(info.CompositeLitType, qualifier)
		}
		if !add(fmt.Sprintf("// Inside composite literal: %s\n", typeName)) {
			return
		}

		// Check if it's a struct literal (underlying might be pointer)
		var isStruct bool
		if info.CompositeLitType != nil {
			currentType := info.CompositeLitType.Underlying()
			if ptr, ok := currentType.(*types.Pointer); ok && ptr.Elem() != nil {
				currentType = ptr.Elem().Underlying()
			}
			// Removed unused 'st' variable here
			_, isStruct = currentType.(*types.Struct)
		}

		if isStruct {
			if !add("//   (Completing struct fields...)\n") {
				return
			}
		}
	}

	// --- Identifier Context ---
	if info.IdentifierAtCursor != nil && !contextAdded {
		contextAdded = true
		identName := info.IdentifierAtCursor.Name
		identTypeStr := "(Type unknown)"
		if info.IdentifierType != nil {
			identTypeStr = fmt.Sprintf("(%s)", types.TypeString(info.IdentifierType, qualifier))
		}
		if !add(fmt.Sprintf("// Identifier context: %s %s\n", identName, identTypeStr)) {
			return
		}
	}

	if !contextAdded {
		add("// General code context.\n")
	}
}

// formatScopeSection formats variables/constants/types in scope, respecting limits.
func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if len(info.VariablesInScope) == 0 {
		return
	}
	if !add("// In Scope:\n") {
		return
	}

	var items []string
	for _, obj := range info.VariablesInScope { // Iterate directly over objects
		// Skip receiver if present
		if info.EnclosingFunc != nil && info.EnclosingFunc.Type() != nil {
			if sig, ok := info.EnclosingFunc.Type().(*types.Signature); ok && sig.Recv() != nil && sig.Recv() == obj {
				continue
			}
		}

		objStr := types.ObjectString(obj, qualifier)
		if objStr == "" {
			objStr = fmt.Sprintf("%s (type: %s)", obj.Name(), types.TypeString(obj.Type(), qualifier))
		}
		objStr = strings.Replace(objStr, " type", "", 1)
		items = append(items, fmt.Sprintf("//   %s\n", objStr))
	}
	sort.Strings(items)

	count := 0
	const maxScopeItems = 35
	for _, item := range items {
		if count >= maxScopeItems {
			addTrunc("scope")
			logger.Debug("Truncated scope list in preamble", "max_items", maxScopeItems)
			return
		}
		if !add(item) {
			return
		}
		count++
	}
}
