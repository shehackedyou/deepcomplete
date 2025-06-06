// deepcomplete/helpers_preamble.go
// Contains helper functions specifically for constructing the LLM prompt preamble.
package deepcomplete

import (
	"context" // Added context
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

// constructPromptPreamble builds the final preamble string by calling specific Analyzer methods.
// This is the primary function called by the completer service.
func constructPromptPreamble(
	analyzer Analyzer, // Use interface type
	info *AstContextInfo, // Still used for file path, version, cursor pos, and specific context nodes
	qualifier types.Qualifier, // Qualifier might still be useful here or derived inside
	logger *slog.Logger,
) string {
	if logger == nil {
		logger = slog.Default()
	}
	preambleLogger := logger.With("op", "constructPromptPreamble")

	var preamble strings.Builder
	const internalPreambleLimit = 8192 // Limit overall preamble construction size
	currentLen := 0
	limitReached := false

	// --- Context for Analyzer calls ---
	// Use a background context for now, or propagate from the original request if available
	ctx := context.Background() // TODO: Propagate original context if possible

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
		preambleLogger.Debug("Internal preamble construction limit reached", "limit", internalPreambleLimit, "current_len", currentLen)
		return false
	}

	// Helper to add truncation marker
	addTruncMarker := func(section string) {
		if limitReached {
			return
		}
		msg := fmt.Sprintf("// ... (%s truncated)\n", section)
		if currentLen+len(msg) < internalPreambleLimit {
			preamble.WriteString(msg)
			currentLen += len(msg)
		}
		limitReached = true
	}

	// --- Build Preamble Sections using Analyzer methods ---
	pkgName := info.PackageName // Get initial package name if available
	if pkgName == "" && info.TargetPackage != nil && info.TargetPackage.Types != nil {
		pkgName = info.TargetPackage.Types.Name()
	}
	if pkgName == "" {
		pkgName = "[unknown]"
	}
	addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), pkgName))

	// Add Imports (still relies on info struct for now)
	if !limitReached {
		formatImportsSection(&preamble, info, addToPreamble, addTruncMarker, preambleLogger)
	}

	// Need FileSet to convert Pos to Line/Column
	fset := info.TargetFileSet
	var cursorLine, cursorCol int = 1, 1 // Default if invalid
	if fset != nil && info.CursorPos.IsValid() {
		posInfo := fset.Position(info.CursorPos)
		cursorLine = posInfo.Line
		cursorCol = posInfo.Column
	} else {
		preambleLogger.Warn("Cannot get accurate line/col for preamble getters: FileSet or CursorPos invalid")
	}

	// Get Enclosing Context Info
	var enclosingCtx *EnclosingContextInfo
	if !limitReached {
		var encErr error
		// Use line/col 0,0 as it doesn't matter for this specific info getter
		enclosingCtx, encErr = analyzer.GetEnclosingContext(ctx, info.FilePath, info.Version, 0, 0)
		if encErr != nil {
			preambleLogger.Warn("Failed to get enclosing context for preamble", "error", encErr)
			addToPreamble("// Error getting enclosing function context.\n")
		} else if enclosingCtx != nil {
			// Pass the retrieved *EnclosingContextInfo
			formatEnclosingFuncSection(&preamble, enclosingCtx, qualifier, addToPreamble, preambleLogger)
		}
	}

	// Get Relevant Comments
	var comments []string
	if !limitReached {
		var commentErr error
		// Pass actual cursor line/col derived from Pos
		comments, commentErr = analyzer.GetRelevantComments(ctx, info.FilePath, info.Version, cursorLine, cursorCol)
		if commentErr != nil {
			preambleLogger.Warn("Failed to get relevant comments for preamble", "error", commentErr)
			addToPreamble("// Error getting relevant comments.\n")
		} else if len(comments) > 0 {
			// Pass the retrieved []string
			formatCommentsSection(&preamble, comments, addToPreamble, addTruncMarker, preambleLogger)
		}
	}

	// Get Scope Info
	var scopeInfo *ScopeInfo
	if !limitReached {
		var scopeErr error
		// Pass actual cursor line/col derived from Pos
		scopeInfo, scopeErr = analyzer.GetScopeInfo(ctx, info.FilePath, info.Version, cursorLine, cursorCol)
		if scopeErr != nil {
			preambleLogger.Warn("Failed to get scope info for preamble", "error", scopeErr)
			addToPreamble("// Error getting scope information.\n")
		} else if scopeInfo != nil && len(scopeInfo.Variables) > 0 {
			// Pass the retrieved *ScopeInfo
			formatScopeSection(&preamble, scopeInfo, qualifier, addToPreamble, addTruncMarker, preambleLogger)
		}
	}

	// Format specific cursor context (still uses info struct)
	if !limitReached {
		formatCursorContextSection(&preamble, info, qualifier, addToPreamble, preambleLogger)
	}

	return preamble.String()
}

// formatImportsSection formats the import list, respecting limits.
// Note: This still relies on AstContextInfo containing Imports.
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if info == nil || len(info.Imports) == 0 {
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

// formatEnclosingFuncSection formats the enclosing function/method info from EnclosingContextInfo.
// Accepts *EnclosingContextInfo.
func formatEnclosingFuncSection(preamble *strings.Builder, encInfo *EnclosingContextInfo, qualifier types.Qualifier, add func(string) bool, logger *slog.Logger) {
	if encInfo == nil {
		return
	} // Nothing to format

	funcOrMethod := "Function"
	receiverStr := ""
	if encInfo.Receiver != "" {
		funcOrMethod = "Method"
		receiverBase := encInfo.Receiver
		if strings.HasPrefix(receiverBase, "*") {
			receiverBase = receiverBase[1:]
		}
		receiverStr = fmt.Sprintf("(%s) ", receiverBase)
	}

	var funcHeader string
	if encInfo.Func != nil { // Prefer type information
		name := encInfo.Func.Name()
		sigStr := types.TypeString(encInfo.Func.Type(), qualifier)
		if encInfo.Receiver != "" && strings.HasPrefix(sigStr, "func(") {
			if firstParen := strings.Index(sigStr, "("); firstParen != -1 {
				if secondParen := strings.Index(sigStr[firstParen+1:], ")"); secondParen != -1 {
					sigStr = "func" + sigStr[firstParen+1+secondParen+1:]
				}
			}
		}
		funcHeader = fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr)
	} else if encInfo.FuncNode != nil { // Fallback to AST
		name := "[anonymous]"
		if encInfo.FuncNode.Name != nil {
			name = encInfo.FuncNode.Name.Name
		}
		paramsStr := "..."
		resultsStr := ""
		if encInfo.FuncNode.Type != nil {
			if encInfo.FuncNode.Type.Params != nil {
				paramsStr = fmt.Sprintf("(%d params)", len(encInfo.FuncNode.Type.Params.List))
			}
			if encInfo.FuncNode.Type.Results != nil {
				resultsStr = fmt.Sprintf(" (%d results)", len(encInfo.FuncNode.Type.Results.List))
			}
		}
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s%s%s\n", funcOrMethod, receiverStr, name, paramsStr, resultsStr)
	} else {
		return
	}
	add(funcHeader)
}

// formatCommentsSection formats relevant comments, respecting limits.
// Accepts comments directly now ( []string ).
func formatCommentsSection(preamble *strings.Builder, comments []string, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if len(comments) == 0 {
		return
	}
	if !add("// Relevant Comments:\n") {
		return
	}

	count := 0
	const maxComments = 7
	for _, c := range comments {
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
// Note: This still relies on AstContextInfo.
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, logger *slog.Logger) {
	if info == nil {
		return
	} // Need info for this section
	if logger == nil {
		logger = slog.Default()
	}
	contextAdded := false

	// --- Function Call Context ---
	if info.CallExpr != nil {
		contextAdded = true
		funcName := "[unknown function]"
		if info.TargetPackage != nil && info.TargetPackage.TypesInfo != nil { // Check if type info is available
			var funcIdent *ast.Ident
			switch fun := info.CallExpr.Fun.(type) {
			case *ast.Ident:
				funcIdent = fun
				funcName = fun.Name
			case *ast.SelectorExpr:
				funcIdent = fun.Sel
				funcName = fun.Sel.Name
			}
			if funcIdent != nil {
				if obj, ok := info.TargetPackage.TypesInfo.Uses[funcIdent]; ok && obj != nil {
					funcName = obj.Name()
				} else if obj, ok := info.TargetPackage.TypesInfo.Defs[funcIdent]; ok && obj != nil {
					funcName = obj.Name()
				}
			}
		} else { // Fallback to AST name if no type info
			switch fun := info.CallExpr.Fun.(type) {
			case *ast.Ident:
				funcName = fun.Name
			case *ast.SelectorExpr:
				if fun.Sel != nil {
					funcName = fun.Sel.Name
				}
			}
		}

		if !add(fmt.Sprintf("// Inside function call: %s (Arg Index %d)\n", funcName, info.CallArgIndex)) {
			return
		}

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
		var isStruct bool
		if info.CompositeLitType != nil {
			currentType := info.CompositeLitType.Underlying()
			if ptr, ok := currentType.(*types.Pointer); ok && ptr.Elem() != nil {
				currentType = ptr.Elem().Underlying()
			}
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
// Accepts *ScopeInfo.
func formatScopeSection(preamble *strings.Builder, scopeInfo *ScopeInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if scopeInfo == nil || len(scopeInfo.Variables) == 0 {
		return
	}
	if !add("// In Scope:\n") {
		return
	}

	var items []string
	for _, obj := range scopeInfo.Variables {
		// TODO: Add logic to skip receiver if needed, potentially requires EnclosingContextInfo here too
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
