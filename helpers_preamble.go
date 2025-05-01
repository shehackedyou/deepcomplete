// deepcomplete/helpers_preamble.go
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
// Conceptual: Could be wrapped by memory cache in the future.
func constructPromptPreamble(
	analyzer *GoPackagesAnalyzer, // Pass down for potential future use (e.g., cache checks within)
	info *AstContextInfo,
	qualifier types.Qualifier,
	logger *slog.Logger,
) string {
	// --- Memory Cache Check (Conceptual - Omitted for now) ---
	// cacheKey := generateCacheKey("preamble", info) // Assumes helpers_cache.go exists
	// cachedPreamble, hit, err := withMemoryCache[string](...)
	// if err != nil { ... }
	// if hit { return cachedPreamble }

	// --- Direct Implementation ---
	return buildPreamble(analyzer, info, qualifier, logger)
}

// buildPreamble constructs the context string sent to the LLM from analysis info.
// Internal helper called by constructPromptPreamble.
// ** MODIFIED: Cycle 3 - Ensure logger is used correctly, improve limit handling **
func buildPreamble(
	analyzer *GoPackagesAnalyzer, // Pass down for potential future use
	info *AstContextInfo,
	qualifier types.Qualifier,
	logger *slog.Logger,
) string {
	if logger == nil {
		logger = slog.Default() // Ensure logger is not nil
	}
	var preamble strings.Builder
	// Use a reasonable internal limit, final truncation happens later in FormatPrompt
	const internalPreambleLimit = 8192
	currentLen := 0
	limitReached := false // Track if limit was hit

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
		logger.Debug("Internal preamble limit reached during construction", "limit", internalPreambleLimit)
		return false
	}

	// Helper to add truncation marker if within limit
	addTruncMarker := func(section string) {
		if limitReached { // Don't add marker if limit was already hit before this section
			return
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
		pkgName = info.TargetPackage.Types.Name() // Get package name from types if available
	}
	if pkgName == "" {
		pkgName = "[unknown]" // Fallback
	}
	addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), pkgName))

	// 2. Add imports section
	formatImportsSection(&preamble, info, addToPreamble, addTruncMarker, logger)

	// 3. Add enclosing function/method section
	formatEnclosingFuncSection(&preamble, info, qualifier, addToPreamble)

	// 4. Add relevant comments section
	formatCommentsSection(&preamble, info, addToPreamble, addTruncMarker, logger)

	// 5. Add specific cursor context (call, selector, etc.) section
	formatCursorContextSection(&preamble, info, qualifier, addToPreamble, logger)

	// 6. Add variables/constants/types in scope section
	formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker, logger)
	// Don't return early here, even if truncated, as scope is often the largest part

	return preamble.String()
}

// formatImportsSection formats the import list, respecting limits.
// Returns false if the internal preamble limit was reached during processing.
// ** MODIFIED: Cycle 3 - Handle nil imports gracefully **
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) bool {
	if len(info.Imports) == 0 {
		return true // No imports, section skipped successfully
	}
	if !add("// Imports:\n") {
		return false // Limit reached just trying to add section header
	}

	count := 0
	maxImports := 20 // Limit number of imports shown for brevity
	for _, imp := range info.Imports {
		if imp == nil || imp.Path == nil {
			logger.Warn("Skipping nil import spec or import path during preamble generation")
			continue // Skip nil imports
		}
		if count >= maxImports {
			addTrunc("imports") // Add truncation marker if limit exceeded
			logger.Debug("Truncated imports list in preamble", "max_imports", maxImports)
			return true // Stop processing imports, but section was added successfully up to this point
		}
		path := imp.Path.Value // Import path (e.g., "fmt")
		name := ""
		if imp.Name != nil {
			name = imp.Name.Name + " " // Named import (e.g., f "fmt")
		}
		line := fmt.Sprintf("//   import %s%s\n", name, path)
		if !add(line) {
			return false // Limit reached while adding an import line
		}
		count++
	}
	return true // Finished adding imports within limits
}

// formatEnclosingFuncSection formats the enclosing function/method info.
// Returns false if the internal preamble limit was reached.
// ** MODIFIED: Cycle 3 - Improve fallback logic **
func formatEnclosingFuncSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool) bool {
	funcOrMethod := "Function"
	receiverStr := ""
	if info.ReceiverType != "" {
		funcOrMethod = "Method"
		receiverStr = fmt.Sprintf("(%s) ", info.ReceiverType)
	}

	var funcHeader string
	if info.EnclosingFunc != nil { // Prefer type information if available
		name := info.EnclosingFunc.Name()
		sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		// Clean up signature string for methods to avoid duplicating receiver
		if info.ReceiverType != "" && strings.HasPrefix(sigStr, "func(") {
			// Remove the receiver part from the signature string as we add it separately
			if firstParen := strings.Index(sigStr, "("); firstParen != -1 {
				if secondParen := strings.Index(sigStr[firstParen+1:], ")"); secondParen != -1 {
					// Construct signature without receiver: "func" + params + results
					sigStr = "func" + sigStr[firstParen+1+secondParen+1:]
				}
			}
		}
		funcHeader = fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr)
	} else if info.EnclosingFuncNode != nil { // Fallback to AST node if type info failed
		name := "[anonymous]"
		if info.EnclosingFuncNode.Name != nil {
			name = info.EnclosingFuncNode.Name.Name
		}
		// Show only basic signature from AST (params/results omitted for brevity/simplicity)
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s(...)\n", funcOrMethod, receiverStr, name)
	} else {
		return true // No enclosing function found, successful no-op
	}
	return add(funcHeader) // Add the formatted header, return success/failure based on limit
}

// formatCommentsSection formats relevant comments, respecting limits.
// Returns false if the internal preamble limit was reached.
func formatCommentsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string), logger *slog.Logger) bool {
	if len(info.CommentsNearCursor) == 0 {
		return true // No comments, section skipped successfully
	}
	if !add("// Relevant Comments:\n") {
		return false // Limit reached adding header
	}

	count := 0
	maxComments := 5 // Limit number of comments shown
	for _, c := range info.CommentsNearCursor {
		if count >= maxComments {
			addTrunc("comments") // Add truncation marker
			logger.Debug("Truncated comments list in preamble", "max_comments", maxComments)
			return true // Stop processing comments, section added successfully up to here
		}
		// Basic cleaning: remove comment markers, trim space
		cleanComment := strings.TrimSpace(strings.TrimPrefix(c, "//"))
		cleanComment = strings.TrimSpace(strings.TrimPrefix(cleanComment, "/*"))
		cleanComment = strings.TrimSpace(strings.TrimSuffix(cleanComment, "*/"))

		if len(cleanComment) > 0 {
			line := fmt.Sprintf("//   %s\n", cleanComment) // Add cleaned comment line
			if !add(line) {
				return false // Limit reached while adding a comment
			}
			count++
		}
	}
	return true // Finished adding comments within limits
}

// formatCursorContextSection formats specific context like calls, selectors, etc.
// Returns false if the internal preamble limit was reached.
// ** MODIFIED: Cycle 3 - Improve handling of unknown types/members and logging **
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, logger *slog.Logger) bool {
	if logger == nil {
		logger = slog.Default() // Ensure logger is not nil
	}
	contextAdded := false // Track if any context was added

	// --- Function Call Context ---
	if info.CallExpr != nil {
		contextAdded = true
		funcName := "[unknown function]"
		// Attempt to get function name from AST
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
			// Add full signature
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
					// Highlight the parameter corresponding to the cursor's argument index
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
					resultStr := types.ObjectString(r, qualifier)
					if resultStr == "" {
						resultStr = fmt.Sprintf("[unnamed %s]", types.TypeString(r.Type(), qualifier))
					}
					if !add(fmt.Sprintf("//     - %s\n", resultStr)) {
						return false
					}
				}
			} else {
				if !add("//   Returns: (none)\n") {
					return false
				}
			}
		} else {
			// Indicate if type analysis failed for the call expression
			logger.Warn("Could not determine function signature for call expression", "function_name", funcName)
			if !add("// Function Signature: (unknown - type analysis failed for call expression)\n") {
				return false
			}
		}
		return true // Handled call expression context successfully (within limits)
	}

	// --- Selector Expression Context ---
	if info.SelectorExpr != nil {
		contextAdded = true
		selName := ""
		if info.SelectorExpr.Sel != nil {
			selName = info.SelectorExpr.Sel.Name
		}
		typeName := "(unknown - type analysis failed for base expression)"
		if info.SelectorExprType != nil {
			typeName = types.TypeString(info.SelectorExprType, qualifier)
		}
		// Check if the selected member is known (useful for diagnostics, less so for preamble maybe)
		fieldOrMethod, _, _ := types.LookupFieldOrMethod(info.SelectorExprType, true, nil, selName)
		isKnownMember := fieldOrMethod != nil
		unknownMemberMsg := ""
		if info.SelectorExprType != nil && !isKnownMember && selName != "" {
			unknownMemberMsg = fmt.Sprintf(" (unknown member '%s')", selName)
		}
		if !add(fmt.Sprintf("// Selector context: expr type = %s%s\n", typeName, unknownMemberMsg)) {
			return false
		}
		// List available members if the base type is known
		if info.SelectorExprType != nil {
			// List members only if the selected member is known OR the selection is incomplete (selName is empty)
			// Avoid listing members if the user typed an explicitly unknown member.
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
				// Add fields section
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
				// Add methods section
				if len(methods) > 0 {
					if !add("//   Available Methods:\n") {
						return false
					}
					sort.Slice(methods, func(i, j int) bool { return methods[i].Name < methods[j].Name })
					for _, method := range methods {
						methodSig := strings.TrimPrefix(method.TypeString, "func") // Remove leading "func"
						if !add(fmt.Sprintf("//     - %s%s\n", method.Name, methodSig)) {
							return false
						}
					}
				}
				// Indicate if no members found or couldn't be determined
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
				logger.Debug("Not listing members because selected member is unknown", "selector", selName, "base_type", typeName)
				if !add("//   (Cannot list members: selected member is unknown)\n") {
					return false
				}
			}
		} else {
			// Base expression type unknown
			logger.Warn("Cannot list members for selector because base expression type is unknown")
			if !add("//   (Cannot list members: type analysis failed for base expression)\n") {
				return false
			}
		}
		return true // Handled selector expression context successfully (within limits)
	}

	// --- Composite Literal Context ---
	if info.CompositeLit != nil {
		contextAdded = true
		typeName := "(unknown - type analysis failed for literal)"
		if info.CompositeLitType != nil {
			typeName = types.TypeString(info.CompositeLitType, qualifier)
		}
		if !add(fmt.Sprintf("// Inside composite literal of type: %s\n", typeName)) {
			return false
		}
		// List missing fields if it's a struct literal
		if info.CompositeLitType != nil {
			var st *types.Struct
			// Get underlying type, handling pointers
			currentType := info.CompositeLitType.Underlying()
			if ptr, ok := currentType.(*types.Pointer); ok {
				if ptr.Elem() != nil {
					currentType = ptr.Elem().Underlying()
				} else {
					logger.Warn("Composite literal pointer type has nil element", "type", info.CompositeLitType.String())
					currentType = nil // Invalid pointer type
				}
			}
			st, _ = currentType.(*types.Struct)

			if st != nil {
				// Find fields already present in the literal
				presentFields := make(map[string]bool)
				for _, elt := range info.CompositeLit.Elts {
					if kv, ok := elt.(*ast.KeyValueExpr); ok {
						if kid, ok := kv.Key.(*ast.Ident); ok {
							presentFields[kid.Name] = true
						}
					}
				}
				// Find exported fields missing from the literal
				var missingFields []string
				for i := 0; i < st.NumFields(); i++ {
					field := st.Field(i)
					if field != nil && field.Exported() && !presentFields[field.Name()] {
						missingFields = append(missingFields, types.ObjectString(field, qualifier))
					}
				}
				// Add missing fields to preamble
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
				// Literal type is known but not a struct
				if !add("//   (Underlying type is not a struct)\n") {
					return false
				}
			}
		} else {
			// Literal type is unknown
			logger.Warn("Cannot determine missing fields for composite literal: type analysis failed")
			if !add("//   (Cannot determine missing fields: type analysis failed)\n") {
				return false
			}
		}
		return true // Handled composite literal context successfully (within limits)
	}

	// --- Identifier Context ---
	if info.IdentifierAtCursor != nil {
		contextAdded = true
		identName := info.IdentifierAtCursor.Name
		identTypeStr := "(Type unknown)"
		if info.IdentifierType != nil {
			identTypeStr = fmt.Sprintf("(Type: %s)", types.TypeString(info.IdentifierType, qualifier))
		} else {
			logger.Warn("Identifier found at cursor, but type is unknown", "identifier", identName)
		}
		if !add(fmt.Sprintf("// Identifier at cursor: %s %s\n", identName, identTypeStr)) {
			return false
		}
		return true // Handled identifier context successfully (within limits)
	}

	// If no specific context was added, maybe add a generic marker?
	if !contextAdded {
		// Optionally add a line indicating no specific context was identified
		// add("// No specific call, selector, or literal context identified at cursor.\n")
	}

	return true // No specific cursor context found or limit reached earlier
}

// formatScopeSection formats variables/constants/types in scope, respecting limits.
// Returns false if the internal preamble limit was reached.
// ** MODIFIED: Cycle 3 - Ensure sorting and truncation work correctly **
func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string), logger *slog.Logger) bool {
	if len(info.VariablesInScope) == 0 {
		return true // No scope variables, section skipped successfully
	}
	if !add("// Variables/Constants/Types in Scope:\n") {
		return false // Limit reached adding header
	}

	// Collect formatted scope items
	var items []string
	for name := range info.VariablesInScope {
		obj := info.VariablesInScope[name]
		objStr := types.ObjectString(obj, qualifier)
		if objStr == "" {
			// Fallback for objects that don't format well (e.g., unnamed results)
			objStr = fmt.Sprintf("%s (type: %s)", obj.Name(), types.TypeString(obj.Type(), qualifier))
		}
		items = append(items, fmt.Sprintf("//   %s\n", objStr))
	}
	sort.Strings(items) // Sort for consistent order

	count := 0
	maxScopeItems := 30 // Limit number of scope items shown
	for _, item := range items {
		if count >= maxScopeItems {
			addTrunc("scope") // Add truncation marker
			logger.Debug("Truncated scope list in preamble", "max_items", maxScopeItems)
			return true // Stop processing scope items, section added successfully up to here
		}
		if !add(item) {
			return false // Limit reached while adding a scope item
		}
		count++
	}
	return true // Finished adding scope items within limits
}
