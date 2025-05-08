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
// This acts as the main entry point for preamble generation.
// Note: Caching for this specific function is omitted for now, as its inputs
// (info struct) are complex and might change frequently. Caching is applied
// to the sub-steps within the analyzer instead.
func constructPromptPreamble(
	analyzer Analyzer, // Use interface type
	info *AstContextInfo,
	qualifier types.Qualifier,
	logger *slog.Logger,
) string {
	// --- Direct Implementation ---
	return buildPreamble(analyzer, info, qualifier, logger) // Pass logger
}

// buildPreamble constructs the context string sent to the LLM from analysis info.
// Internal helper called by constructPromptPreamble.
func buildPreamble(
	analyzer Analyzer, // Use interface type
	info *AstContextInfo,
	qualifier types.Qualifier,
	logger *slog.Logger,
) string {
	if logger == nil {
		logger = slog.Default() // Ensure logger is not nil
	}
	var preamble strings.Builder
	// Use a reasonable internal limit; final truncation happens later in FormatPrompt
	// This limit prevents excessively long intermediate preambles during construction.
	const internalPreambleLimit = 8192
	currentLen := 0
	limitReached := false // Track if limit was hit during construction

	// Helper to add string to preamble if within limit
	addToPreamble := func(s string) bool {
		if limitReached {
			return false // Stop adding if limit already hit
		}
		if currentLen+len(s) < internalPreambleLimit {
			preamble.WriteString(s)
			currentLen += len(s)
			return true
		}
		// Limit reached *now*
		limitReached = true
		logger.Debug("Internal preamble construction limit reached", "limit", internalPreambleLimit, "current_len", currentLen)
		return false
	}

	// Helper to add truncation marker if the limit wasn't *already* hit before this section
	addTruncMarker := func(section string) {
		if limitReached {
			return // Don't add marker if limit was already hit before this section started
		}
		msg := fmt.Sprintf("//   ... (%s truncated)\n", section)
		// Try to add the marker itself, respecting the limit
		if currentLen+len(msg) < internalPreambleLimit {
			preamble.WriteString(msg)
			currentLen += len(msg)
		}
		limitReached = true // Mark limit as reached even if marker couldn't be added
	}

	// --- Build Preamble Sections ---
	// 1. Add file/package context
	pkgName := info.PackageName
	if pkgName == "" && info.TargetPackage != nil && info.TargetPackage.Types != nil {
		pkgName = info.TargetPackage.Types.Name()
	}
	if pkgName == "" {
		pkgName = "[unknown]"
	}
	// Use addToPreamble to respect limit
	addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), pkgName))

	// 2. Add imports section
	formatImportsSection(&preamble, info, addToPreamble, addTruncMarker, logger) // Pass logger

	// 3. Add enclosing function/method section
	formatEnclosingFuncSection(&preamble, info, qualifier, addToPreamble, logger) // Pass logger

	// 4. Add relevant comments section
	formatCommentsSection(&preamble, info, addToPreamble, addTruncMarker, logger) // Pass logger

	// 5. Add specific cursor context (call, selector, etc.) section
	formatCursorContextSection(&preamble, info, qualifier, addToPreamble, logger) // Pass logger

	// 6. Add variables/constants/types in scope section
	formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker, logger) // Pass logger

	return preamble.String()
}

// formatImportsSection formats the import list, respecting limits.
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if len(info.Imports) == 0 {
		return // No imports, section skipped successfully
	}
	if !add("// Imports:\n") {
		return // Limit reached just trying to add section header
	}

	count := 0
	const maxImports = 20 // Limit number of imports shown for brevity
	for _, imp := range info.Imports {
		if imp == nil || imp.Path == nil {
			logger.Warn("Skipping nil import spec or import path during preamble generation")
			continue
		}
		if count >= maxImports {
			addTrunc("imports") // Add truncation marker if limit exceeded
			logger.Debug("Truncated imports list in preamble", "max_imports", maxImports)
			return // Stop processing imports
		}
		path := imp.Path.Value // Import path (e.g., "fmt")
		name := ""
		if imp.Name != nil {
			name = imp.Name.Name + " " // Named import (e.g., f "fmt")
		}
		line := fmt.Sprintf("//   import %s%s\n", name, path)
		if !add(line) {
			return // Limit reached while adding an import line
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
		receiverStr = fmt.Sprintf("(%s) ", info.ReceiverType)
	}

	var funcHeader string
	if info.EnclosingFunc != nil { // Prefer type information
		name := info.EnclosingFunc.Name()
		sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		// Clean up signature string for methods
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
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s(...)\n", funcOrMethod, receiverStr, name)
	} else {
		return // No enclosing function found
	}
	add(funcHeader) // Add the formatted header, respecting limit via 'add'
}

// formatCommentsSection formats relevant comments, respecting limits.
func formatCommentsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if len(info.CommentsNearCursor) == 0 {
		return // No comments, section skipped successfully
	}
	if !add("// Relevant Comments:\n") {
		return // Limit reached adding header
	}

	count := 0
	const maxComments = 5 // Limit number of comments shown
	for _, c := range info.CommentsNearCursor {
		if count >= maxComments {
			addTrunc("comments") // Add truncation marker
			logger.Debug("Truncated comments list in preamble", "max_comments", maxComments)
			return // Stop processing comments
		}
		// Basic cleaning
		cleanComment := strings.TrimSpace(strings.TrimPrefix(c, "//"))
		cleanComment = strings.TrimSpace(strings.TrimPrefix(cleanComment, "/*"))
		cleanComment = strings.TrimSpace(strings.TrimSuffix(cleanComment, "*/"))

		if len(cleanComment) > 0 {
			line := fmt.Sprintf("//   %s\n", cleanComment)
			if !add(line) {
				return // Limit reached while adding a comment
			}
			count++
		}
	}
}

// formatCursorContextSection formats specific context like calls, selectors, etc.
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, logger *slog.Logger) {
	if logger == nil {
		logger = slog.Default() // Ensure logger is not nil
	}
	contextAdded := false // Track if any specific context was added

	// --- Function Call Context ---
	if info.CallExpr != nil {
		contextAdded = true
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
			return
		}

		if sig := info.CallExprFuncType; sig != nil {
			if !add(fmt.Sprintf("// Function Signature: %s\n", types.TypeString(sig, qualifier))) {
				return
			}
			params := sig.Params()
			if params != nil && params.Len() > 0 {
				if !add("//   Parameters:\n") {
					return
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
					paramStr := types.ObjectString(p, qualifier)
					if paramStr == "" {
						paramStr = fmt.Sprintf("[unnamed %s]", types.TypeString(p.Type(), qualifier))
					}
					if !add(fmt.Sprintf("//     - %s%s\n", paramStr, highlight)) {
						return
					}
				}
			} else {
				if !add("//   Parameters: (none)\n") {
					return
				}
			}
			results := sig.Results()
			if results != nil && results.Len() > 0 {
				if !add("//   Returns:\n") {
					return
				}
				for i := 0; i < results.Len(); i++ {
					r := results.At(i)
					if r == nil {
						continue
					}
					resultStr := types.ObjectString(r, qualifier)
					if resultStr == "" {
						resultStr = fmt.Sprintf("[unnamed %s]", types.TypeString(r.Type(), qualifier))
					}
					if !add(fmt.Sprintf("//     - %s\n", resultStr)) {
						return
					}
				}
			} else {
				if !add("//   Returns: (none)\n") {
					return
				}
			}
		} else {
			logger.Warn("Could not determine function signature for call expression", "function_name", funcName)
			if !add("// Function Signature: (unknown - type analysis failed for call expression)\n") {
				return
			}
		}
		// Allow fall-through to other contexts if needed, though unlikely inside a call
	}

	// --- Selector Expression Context ---
	if info.SelectorExpr != nil && !contextAdded { // Check !contextAdded to avoid overlap if cursor is somehow in both
		contextAdded = true
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
			return
		}

		if info.SelectorExprType != nil {
			if isKnownMember || selName == "" {
				members := listTypeMembers(info.SelectorExprType, info.SelectorExpr.X, qualifier, logger) // Assumes helpers_analysis_steps.go exists
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
						return
					}
					sort.Slice(fields, func(i, j int) bool { return fields[i].Name < fields[j].Name })
					for _, field := range fields {
						if !add(fmt.Sprintf("//     - %s %s\n", field.Name, field.TypeString)) {
							return
						}
					}
				}
				if len(methods) > 0 {
					if !add("//   Available Methods:\n") {
						return
					}
					sort.Slice(methods, func(i, j int) bool { return methods[i].Name < methods[j].Name })
					for _, method := range methods {
						methodSig := strings.TrimPrefix(method.TypeString, "func")
						if !add(fmt.Sprintf("//     - %s%s\n", method.Name, methodSig)) {
							return
						}
					}
				}
				if len(fields) == 0 && len(methods) == 0 {
					msg := "//   (No exported fields or methods found)\n"
					if members == nil {
						msg = "//   (Could not determine members)\n"
					}
					if !add(msg) {
						return
					}
				}
			} else { // Selected member is explicitly unknown
				logger.Debug("Not listing members because selected member is unknown", "selector", selName, "base_type", typeName)
				if !add("//   (Cannot list members: selected member is unknown)\n") {
					return
				}
			}
		} else {
			logger.Warn("Cannot list members for selector because base expression type is unknown")
			if !add("//   (Cannot list members: type analysis failed for base expression)\n") {
				return
			}
		}
		// Allow fall-through
	}

	// --- Composite Literal Context ---
	if info.CompositeLit != nil && !contextAdded { // Check !contextAdded
		contextAdded = true
		typeName := "(unknown - type analysis failed for literal)"
		if info.CompositeLitType != nil {
			typeName = types.TypeString(info.CompositeLitType, qualifier)
		}
		if !add(fmt.Sprintf("// Inside composite literal of type: %s\n", typeName)) {
			return
		}

		if info.CompositeLitType != nil {
			var st *types.Struct
			currentType := info.CompositeLitType.Underlying()
			if ptr, ok := currentType.(*types.Pointer); ok {
				if ptr.Elem() != nil {
					currentType = ptr.Elem().Underlying()
				} else {
					logger.Warn("Composite literal pointer type has nil element", "type", info.CompositeLitType.String())
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
						return
					}
					sort.Strings(missingFields)
					for _, fieldStr := range missingFields {
						if !add(fmt.Sprintf("//     - %s\n", fieldStr)) {
							return
						}
					}
				} else {
					if !add("//   (All exported fields may be present or none missing)\n") {
						return
					}
				}
			} else {
				if !add("//   (Underlying type is not a struct)\n") {
					return
				}
			}
		} else {
			logger.Warn("Cannot determine missing fields for composite literal: type analysis failed")
			if !add("//   (Cannot determine missing fields: type analysis failed)\n") {
				return
			}
		}
		// Allow fall-through
	}

	// --- Identifier Context ---
	if info.IdentifierAtCursor != nil && !contextAdded { // Check !contextAdded
		contextAdded = true
		identName := info.IdentifierAtCursor.Name
		identTypeStr := "(Type unknown)"
		if info.IdentifierType != nil {
			identTypeStr = fmt.Sprintf("(Type: %s)", types.TypeString(info.IdentifierType, qualifier))
		} else {
			logger.Warn("Identifier found at cursor, but type is unknown", "identifier", identName)
		}
		if !add(fmt.Sprintf("// Identifier at cursor: %s %s\n", identName, identTypeStr)) {
			return
		}
		// Allow fall-through
	}

	// If no specific context was added, maybe add a generic marker?
	if !contextAdded {
		add("// No specific call, selector, or literal context identified at cursor.\n")
	}
}

// formatScopeSection formats variables/constants/types in scope, respecting limits.
func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string), logger *slog.Logger) {
	if len(info.VariablesInScope) == 0 {
		return // No scope variables, section skipped successfully
	}
	if !add("// Variables/Constants/Types in Scope:\n") {
		return // Limit reached adding header
	}

	var items []string
	for name := range info.VariablesInScope {
		obj := info.VariablesInScope[name]
		objStr := types.ObjectString(obj, qualifier)
		if objStr == "" {
			// Fallback formatting if ObjectString fails
			objStr = fmt.Sprintf("%s (type: %s)", obj.Name(), types.TypeString(obj.Type(), qualifier))
		}
		items = append(items, fmt.Sprintf("//   %s\n", objStr))
	}
	sort.Strings(items) // Sort for consistent order

	count := 0
	const maxScopeItems = 30 // Limit number of scope items shown
	for _, item := range items {
		if count >= maxScopeItems {
			addTrunc("scope") // Add truncation marker
			logger.Debug("Truncated scope list in preamble", "max_items", maxScopeItems)
			return // Stop processing scope items
		}
		if !add(item) {
			return // Limit reached while adding a scope item
		}
		count++
	}
}
