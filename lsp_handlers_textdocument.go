// deepcomplete/lsp_handlers_textdocument.go
// Contains LSP method handlers related to text document synchronization and language features
// (didOpen, didChange, didClose, completion, hover, definition, codeAction, signatureHelp).
package deepcomplete

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/types" // Needed for signature help formatting
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sourcegraph/jsonrpc2"
)

// ============================================================================
// LSP Text Document Method Handlers
// ============================================================================

// handleDidOpen handles the 'textDocument/didOpen' notification.
func (s *Server) handleDidOpen(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidOpenTextDocumentParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	version := params.TextDocument.Version
	content := []byte(params.TextDocument.Text)
	openLogger := logger.With("uri", uri, "version", version, "size", len(content))
	openLogger.Info("Handling textDocument/didOpen")

	s.filesMu.Lock()
	s.files[uri] = &OpenFile{
		URI:     uri,
		Content: content,
		Version: version,
	}
	s.filesMu.Unlock()

	absPath, pathErr := ValidateAndGetFilePath(string(uri), openLogger)
	if pathErr != nil {
		openLogger.Error("Invalid URI in didOpen, cannot trigger diagnostics", "error", pathErr)
		s.sendShowMessage(MessageTypeError, fmt.Sprintf("Invalid document URI: %v", pathErr))
		return nil, nil
	}

	go s.triggerDiagnostics(uri, version, content, absPath, openLogger)
	return nil, nil
}

// handleDidChange handles the 'textDocument/didChange' notification.
func (s *Server) handleDidChange(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidChangeTextDocumentParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	version := params.TextDocument.Version
	changeLogger := logger.With("uri", uri, "new_version", version)

	if len(params.ContentChanges) == 0 {
		changeLogger.Warn("Received didChange notification with no content changes")
		return nil, nil
	}
	newContent := []byte(params.ContentChanges[len(params.ContentChanges)-1].Text)
	changeLogger.Info("Handling textDocument/didChange", "new_size", len(newContent))

	absPath, pathErr := ValidateAndGetFilePath(string(uri), changeLogger)
	if pathErr != nil {
		changeLogger.Error("Invalid URI in didChange", "error", pathErr)
		s.sendShowMessage(MessageTypeError, fmt.Sprintf("Invalid document URI: %v", pathErr))
		return nil, nil
	}

	s.filesMu.Lock()
	currentFile, exists := s.files[uri]
	shouldUpdate := !exists || version > currentFile.Version
	if shouldUpdate {
		s.files[uri] = &OpenFile{
			URI:     uri,
			Content: newContent,
			Version: version,
		}
		changeLogger.Debug("Updated file cache")
	} else {
		changeLogger.Warn("Ignoring out-of-order didChange notification", "received_version", version, "current_version", currentFile.Version)
	}
	s.filesMu.Unlock()

	if shouldUpdate && s.completer.analyzer != nil {
		if err := s.completer.InvalidateMemoryCacheForURI(string(uri), version); err != nil {
			changeLogger.Warn("Failed to invalidate memory cache on didChange", "error", err)
		}

		baseName := filepath.Base(absPath)
		if baseName == "go.mod" || baseName == "go.sum" {
			dir := filepath.Dir(absPath)
			changeLogger.Info("Detected change in go.mod/go.sum, invalidating disk cache", "file", baseName, "dir", dir)
			if err := s.completer.InvalidateAnalyzerCache(dir); err != nil {
				changeLogger.Warn("Failed to invalidate disk cache on didChange", "dir", dir, "error", err)
			}
		} else {
			changeLogger.Debug("Skipping disk cache invalidation for non-go.mod/go.sum file change", "file", baseName)
		}
	}

	go s.triggerDiagnostics(uri, version, newContent, absPath, changeLogger)
	return nil, nil
}

// handleDidClose handles the 'textDocument/didClose' notification.
func (s *Server) handleDidClose(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidCloseTextDocumentParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	closeLogger := logger.With("uri", uri)
	closeLogger.Info("Handling textDocument/didClose")

	s.filesMu.Lock()
	delete(s.files, uri)
	s.filesMu.Unlock()

	s.publishDiagnostics(uri, nil, []LspDiagnostic{}, closeLogger)

	if s.completer.analyzer != nil {
		if err := s.completer.InvalidateMemoryCacheForURI(string(uri), 0); err != nil {
			closeLogger.Warn("Failed to invalidate memory cache on didClose", "error", err)
		}
	}
	return nil, nil
}

// handleCompletion generates code completions based on the current document state and cursor position.
func (s *Server) handleCompletion(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params CompletionParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	lspPos := params.Position
	completionLogger := logger.With("uri", uri, "lsp_line", lspPos.Line, "lsp_char", lspPos.Character, "req_id", req.ID)

	triggerKind := CompletionTriggerKindInvoked
	triggerChar := ""
	if params.Context != nil {
		triggerKind = params.Context.TriggerKind
		triggerChar = params.Context.TriggerCharacter
		completionLogger = completionLogger.With("trigger_kind", triggerKind)
		if triggerChar != "" {
			completionLogger = completionLogger.With("trigger_char", triggerChar)
		}
	}
	completionLogger.Info("Handling textDocument/completion")

	s.filesMu.RLock()
	fileData, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		completionLogger.Warn("Completion request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	line, col, _, posErr := LspPositionToBytePosition(fileData.Content, lspPos, completionLogger)
	if posErr != nil {
		completionLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
	}
	completionLogger = completionLogger.With("go_line", line, "go_col", col)

	absPath, pathErr := ValidateAndGetFilePath(string(uri), completionLogger)
	if pathErr != nil {
		completionLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	// --- Get Completion Text ---
	completionCtx, cancelCompletion := context.WithTimeout(ctx, 10*time.Second)
	defer cancelCompletion()

	var completionBuffer bytes.Buffer
	completionErr := s.completer.GetCompletionStreamFromFile(completionCtx, absPath, fileData.Version, line, col, &completionBuffer)

	select {
	case <-completionCtx.Done():
		completionLogger.Info("Completion request cancelled or timed out during LLM call", "error", completionCtx.Err())
		if errors.Is(completionCtx.Err(), context.DeadlineExceeded) {
			return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
		}
		return nil, &jsonrpc2.Error{Code: int64(JsonRpcRequestCancelled), Message: "Completion request cancelled"}
	default:
	}

	if completionErr != nil {
		if errors.Is(completionErr, ErrOllamaUnavailable) {
			completionLogger.Error("Ollama unavailable", "error", completionErr)
			s.sendShowMessage(MessageTypeError, fmt.Sprintf("Completion backend error: %v", completionErr))
			return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
		}
		if errors.Is(completionErr, ErrAnalysisFailed) {
			completionLogger.Warn("Code analysis failed during completion, proceeding without specific kind", "error", completionErr)
		} else {
			completionLogger.Error("Failed to get completion stream", "error", completionErr)
			return nil, fmt.Errorf("completion failed: %w", completionErr)
		}
	}

	completionText := strings.TrimSpace(completionBuffer.String())
	if completionText == "" {
		completionLogger.Info("Completion successful but result is empty")
		return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
	}

	completionLogger.Info("Completion successful", "completion_length", len(completionText))

	// --- Determine Completion Kind (Best Effort) ---
	completionKind := CompletionItemKindSnippet
	currentConfig := s.completer.GetCurrentConfig()
	if currentConfig.UseAst {
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 2*time.Second)
		defer cancelAnalysis()
		// Use GetIdentifierInfo for potentially faster kind determination
		identInfo, kindAnalysisErr := s.completer.analyzer.GetIdentifierInfo(analysisCtx, absPath, fileData.Version, line, col)
		if kindAnalysisErr != nil {
			completionLogger.Warn("Analysis for completion kind failed, using default kind", "error", kindAnalysisErr)
		} else if identInfo != nil && identInfo.Object != nil { // Check if identifier was found
			completionKind = mapTypeToCompletionKind(identInfo.Object) // Use refined mapping
			completionLogger.Debug("Determined completion kind from analysis", "kind", completionKind, "identifier", identInfo.Name)
		} else {
			completionLogger.Debug("No specific identifier found at cursor for kind mapping, using default.")
		}
	} else {
		completionLogger.Debug("AST analysis disabled, using default completion kind.")
	}

	// --- Determine Insert Text Format ---
	insertTextFormat := PlainTextFormat
	insertText := completionText
	if s.clientCaps.TextDocument != nil &&
		s.clientCaps.TextDocument.Completion != nil &&
		s.clientCaps.TextDocument.Completion.CompletionItem != nil &&
		s.clientCaps.TextDocument.Completion.CompletionItem.SnippetSupport {
		insertTextFormat = SnippetFormat
		insertText = completionText + "${0}"
		completionLogger.Debug("Using Snippet format for completion item")
	} else {
		completionLogger.Debug("Using PlainText format for completion item")
	}

	// --- Create Completion Item ---
	item := CompletionItem{
		Label:            strings.Split(completionText, "\n")[0],
		InsertText:       insertText,
		InsertTextFormat: insertTextFormat,
		Kind:             completionKind,
		Detail:           "DeepComplete Suggestion",
	}

	return CompletionList{
		IsIncomplete: false,
		Items:        []CompletionItem{item},
	}, nil
}

// handleHover generates hover information using the GetIdentifierInfo method.
func (s *Server) handleHover(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params HoverParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	lspPos := params.Position
	hoverLogger := logger.With("uri", uri, "lsp_line", lspPos.Line, "lsp_char", lspPos.Character, "req_id", req.ID)
	hoverLogger.Info("Handling textDocument/hover")

	s.filesMu.RLock()
	fileData, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		hoverLogger.Warn("Hover request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	line, col, _, posErr := LspPositionToBytePosition(fileData.Content, lspPos, hoverLogger)
	if posErr != nil {
		hoverLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil
	}
	hoverLogger = hoverLogger.With("go_line", line, "go_col", col)

	absPath, pathErr := ValidateAndGetFilePath(string(uri), hoverLogger)
	if pathErr != nil {
		hoverLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	// Use the specific GetIdentifierInfo method
	identInfo, analysisErr := s.completer.analyzer.GetIdentifierInfo(analysisCtx, absPath, fileData.Version, line, col)

	select {
	case <-analysisCtx.Done():
		hoverLogger.Info("Hover analysis cancelled or timed out", "error", analysisCtx.Err())
		if errors.Is(analysisCtx.Err(), context.DeadlineExceeded) {
			return nil, nil
		}
		return nil, &jsonrpc2.Error{Code: int64(JsonRpcRequestCancelled), Message: "Hover request cancelled"}
	default:
	}

	if analysisErr != nil {
		if !errors.Is(analysisErr, ErrAnalysisFailed) && !errors.Is(analysisErr, ErrPositionConversion) {
			hoverLogger.Error("Critical error during identifier analysis for hover", "error", analysisErr)
		} else {
			hoverLogger.Warn("Analysis for hover encountered non-critical errors", "error", analysisErr)
		}
		return nil, nil
	}

	if identInfo == nil || identInfo.Object == nil {
		hoverLogger.Debug("No identifier found at cursor position for hover")
		return nil, nil
	}

	// Format hover content using the retrieved IdentifierInfo
	hoverContent := formatObjectForHover(identInfo, hoverLogger) // Pass IdentifierInfo
	if hoverContent == "" {
		hoverLogger.Debug("No hover content generated for identifier", "identifier", identInfo.Name)
		return nil, nil
	}

	// Determine hover range using the identifier node from IdentifierInfo
	var hoverRange *LSPRange
	if identInfo.FileSet != nil && identInfo.IdentNode != nil {
		// Pass the file content stored in identInfo
		lspRange, rangeErr := nodeRangeToLSPRange(identInfo.FileSet, identInfo.IdentNode, identInfo.Content, hoverLogger)
		if rangeErr == nil {
			hoverRange = lspRange
		} else {
			hoverLogger.Warn("Could not determine range for hover identifier", "error", rangeErr)
		}
	}

	markupKind := MarkupKindPlainText
	if s.clientCaps.TextDocument != nil && s.clientCaps.TextDocument.Hover != nil {
		for _, kind := range s.clientCaps.TextDocument.Hover.ContentFormat {
			if kind == MarkupKindMarkdown {
				markupKind = MarkupKindMarkdown
				break
			}
		}
	}

	hoverLogger.Info("Hover information generated successfully", "identifier", identInfo.Name, "markup", markupKind)
	return HoverResult{
		Contents: MarkupContent{Kind: markupKind, Value: hoverContent},
		Range:    hoverRange,
	}, nil
}

// handleDefinition finds the definition location using the GetIdentifierInfo method.
func (s *Server) handleDefinition(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DefinitionParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	lspPos := params.Position
	defLogger := logger.With("uri", uri, "lsp_line", lspPos.Line, "lsp_char", lspPos.Character, "req_id", req.ID)
	defLogger.Info("Handling textDocument/definition")

	s.filesMu.RLock()
	// Get file data needed for position conversion and potential content read
	fileData, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		defLogger.Warn("Definition request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	line, col, _, posErr := LspPositionToBytePosition(fileData.Content, lspPos, defLogger)
	if posErr != nil {
		defLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil
	}
	defLogger = defLogger.With("go_line", line, "go_col", col)

	absPath, pathErr := ValidateAndGetFilePath(string(uri), defLogger)
	if pathErr != nil {
		defLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	// Use the specific GetIdentifierInfo method
	identInfo, analysisErr := s.completer.analyzer.GetIdentifierInfo(analysisCtx, absPath, fileData.Version, line, col)

	select {
	case <-analysisCtx.Done():
		defLogger.Info("Definition analysis cancelled or timed out", "error", analysisCtx.Err())
		if errors.Is(analysisCtx.Err(), context.DeadlineExceeded) {
			return nil, nil
		}
		return nil, &jsonrpc2.Error{Code: int64(JsonRpcRequestCancelled), Message: "Definition request cancelled"}
	default:
	}

	if analysisErr != nil {
		if !errors.Is(analysisErr, ErrAnalysisFailed) && !errors.Is(analysisErr, ErrPositionConversion) {
			defLogger.Error("Critical error during identifier analysis for definition", "error", analysisErr)
		} else {
			defLogger.Warn("Analysis for definition encountered non-critical errors", "error", analysisErr)
		}
		return nil, nil
	}

	if identInfo == nil || identInfo.Object == nil || !identInfo.Object.Pos().IsValid() {
		defLogger.Debug("No identifier or valid definition position found at cursor")
		return nil, nil
	}

	obj := identInfo.Object
	defPos := obj.Pos()                            // token.Pos
	defFileToken := identInfo.FileSet.File(defPos) // Use FileSet from IdentifierInfo
	if defFileToken == nil {
		defLogger.Error("Could not find token.File for definition position", "identifier", obj.Name(), "pos", defPos)
		return nil, nil
	}

	// Read content of the definition file only if needed for conversion
	var defFileContent []byte
	var readErr error
	defFileName := defFileToken.Name()
	if defFileName != absPath { // Check if definition is in a different file
		defFileContent, readErr = os.ReadFile(defFileName)
		if readErr != nil {
			defLogger.Error("Failed to read definition file content", "path", defFileName, "error", readErr)
			s.sendShowMessage(MessageTypeWarning, fmt.Sprintf("Could not read definition file: %s", defFileName))
			return nil, nil
		}
	} else {
		// Use the content already held by the server for the current file
		defFileContent = fileData.Content // Use content from the file map
	}

	// Convert the definition token.Pos to an LSP Location
	location, locErr := tokenPosToLSPLocation(defFileToken, defPos, defFileContent, defLogger)
	if locErr != nil {
		defLogger.Error("Failed to convert definition position to LSP Location", "identifier", obj.Name(), "error", locErr)
		return nil, nil
	}

	defLogger.Info("Definition found", "identifier", obj.Name(), "location_uri", location.URI, "location_line", location.Range.Start.Line)
	return []Location{*location}, nil // Return as slice
}

// handleCodeAction handles the 'textDocument/codeAction' request.
// Provides basic quick fixes based on diagnostics.
func (s *Server) handleCodeAction(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params CodeActionParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	actionLogger := logger.With("uri", uri, "range", params.Range, "req_id", req.ID)
	actionLogger.Info("Handling textDocument/codeAction")

	var codeActions []any // Can be Command or CodeAction

	// Get current file content for potential edits
	s.filesMu.RLock()
	_, ok := s.files[uri] // Check if file exists
	s.filesMu.RUnlock()
	if !ok {
		actionLogger.Warn("CodeAction request for unknown file")
		return nil, nil // Return empty list if file not found
	}

	// Iterate through diagnostics provided in the context
	for _, diag := range params.Context.Diagnostics {
		actionLogger.Debug("Considering diagnostic for code action", "diagnostic_code", diag.Code, "diagnostic_source", diag.Source, "diagnostic_message", diag.Message)

		// Example: Offer a quick fix for unresolved identifiers
		if diag.Source == DiagSourceAnalyzer && diag.Code == DiagCodeUnresolvedIdentifier {
			// TODO: Implement actual fix logic (e.g., suggest imports, create variable)
			action := CodeAction{
				Title:       fmt.Sprintf("Placeholder: Fix unresolved identifier '%s'", diag.Message[len("unresolved identifier: "):]),
				Kind:        CodeActionKindQuickFix,
				Diagnostics: []LspDiagnostic{diag},
			}
			codeActions = append(codeActions, action)
			actionLogger.Debug("Added placeholder quickfix for unresolved identifier")
		}
		// Example: Offer quick fix for basic unused variable check
		if diag.Source == DiagSourceAnalyzer && diag.Code == DiagCodeUnusedVariable {
			var varName string
			parts := strings.SplitN(diag.Message, "'", 3)
			if len(parts) == 3 {
				varName = parts[1]
			}

			actionTitle := "Remove unused variable"
			if varName != "" {
				actionTitle = fmt.Sprintf("Remove unused variable '%s'", varName)
			}

			// Create a simple edit to remove the diagnostic range (basic removal)
			fixEdit := &WorkspaceEdit{
				Changes: map[DocumentURI][]TextEdit{
					uri: {{Range: diag.Range, NewText: ""}},
				},
			}

			action := CodeAction{
				Title:       actionTitle,
				Kind:        CodeActionKindQuickFix,
				Diagnostics: []LspDiagnostic{diag},
				Edit:        fixEdit,
				IsPreferred: true, // Mark as preferred fix
			}
			codeActions = append(codeActions, action)
			actionLogger.Debug("Added quickfix for unused variable")
		}
		// Add more checks for other diagnostic codes/sources here
	}

	if len(codeActions) == 0 {
		actionLogger.Debug("No code actions generated for the given context")
		return nil, nil
	}

	actionLogger.Info("Returning code actions", "count", len(codeActions))
	return codeActions, nil
}

// handleSignatureHelp handles the 'textDocument/signatureHelp' request.
// Attempts to provide signature help based on enclosing call expression.
func (s *Server) handleSignatureHelp(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params SignatureHelpParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	lspPos := params.Position
	sigLogger := logger.With("uri", uri, "lsp_line", lspPos.Line, "lsp_char", lspPos.Character, "req_id", req.ID)
	sigLogger.Info("Handling textDocument/signatureHelp")

	s.filesMu.RLock()
	fileData, ok := s.files[uri]
	s.filesMu.RUnlock()
	if !ok {
		sigLogger.Warn("SignatureHelp request for unknown file")
		return nil, nil // Return null if file not found
	}

	line, col, _, posErr := LspPositionToBytePosition(fileData.Content, lspPos, sigLogger)
	if posErr != nil {
		sigLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil
	}
	sigLogger = sigLogger.With("go_line", line, "go_col", col)

	absPath, pathErr := ValidateAndGetFilePath(string(uri), sigLogger)
	if pathErr != nil {
		sigLogger.Error("Invalid file URI", "error", pathErr)
		return nil, nil // Don't return error to client, just null result
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 5*time.Second) // Shorter timeout for signature help
	defer cancel()

	// Use Analyze for now, as GetEnclosingContext doesn't return everything needed yet
	// TODO: Refactor Analyze further or add a specific GetCallContext method
	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, fileData.Version, line, col)

	select {
	case <-analysisCtx.Done():
		sigLogger.Info("SignatureHelp analysis cancelled or timed out", "error", analysisCtx.Err())
		return nil, nil
	default:
	}

	if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
		sigLogger.Warn("Analysis for signature help encountered critical errors", "error", analysisErr)
		return nil, nil // Return nil on critical analysis errors
	}
	if analysisInfo == nil {
		sigLogger.Warn("Analysis returned nil info for signature help")
		return nil, nil
	}
	// Log non-fatal errors but proceed
	if analysisErr != nil {
		sigLogger.Warn("Analysis for signature help encountered non-fatal errors", "error", analysisErr)
	}

	// Check if the cursor is inside a call expression
	if analysisInfo.CallExpr == nil || analysisInfo.CallExprFuncType == nil {
		sigLogger.Debug("Cursor not inside a known call expression")
		return nil, nil
	}

	sig := analysisInfo.CallExprFuncType
	paramsTuple := sig.Params()
	resultsTuple := sig.Results() // Get results as well

	// Format parameters
	var parameters []ParameterInformation
	if paramsTuple != nil {
		for i := 0; i < paramsTuple.Len(); i++ {
			p := paramsTuple.At(i)
			if p != nil {
				// Use a qualifier relative to the package where the function is defined
				qualifier := types.RelativeTo(analysisInfo.TargetPackage.Types)
				if p.Pkg() != nil && p.Pkg() != analysisInfo.TargetPackage.Types {
					qualifier = types.RelativeTo(p.Pkg()) // Adjust qualifier if param is from different pkg? Less common for params.
				}
				paramLabel := types.TypeString(p.Type(), qualifier)
				if p.Name() != "" {
					paramLabel = p.Name() + " " + paramLabel
				}
				parameters = append(parameters, ParameterInformation{
					Label: paramLabel,
					// TODO: Add documentation lookup for parameters if possible
				})
			}
		}
	}

	// Format signature label (function name + params + results)
	var sigLabelBuilder strings.Builder
	// Get function name (prefer resolved object name)
	funcName := "(unknown)"
	if analysisInfo.IdentifierObject != nil { // Use resolved object name if available
		funcName = analysisInfo.IdentifierObject.Name()
	} else { // Fallback to AST name
		switch fun := analysisInfo.CallExpr.Fun.(type) {
		case *ast.Ident:
			funcName = fun.Name
		case *ast.SelectorExpr:
			if fun.Sel != nil {
				funcName = fun.Sel.Name
			}
		default:
			sigLogger.Warn("Could not determine function name from AST for signature help")
		}
	}
	sigLabelBuilder.WriteString(funcName)
	sigLabelBuilder.WriteString("(")
	for i, p := range parameters {
		if i > 0 {
			sigLabelBuilder.WriteString(", ")
		}
		// Assuming label is string for now, might need adjustment if label offsets are used
		if labelStr, ok := p.Label.(string); ok {
			sigLabelBuilder.WriteString(labelStr)
		}
	}
	sigLabelBuilder.WriteString(")")

	// Add results to label
	if resultsTuple != nil && resultsTuple.Len() > 0 {
		sigLabelBuilder.WriteString(" ")
		if resultsTuple.Len() > 1 {
			sigLabelBuilder.WriteString("(")
		}
		for i := 0; i < resultsTuple.Len(); i++ {
			r := resultsTuple.At(i)
			if i > 0 {
				sigLabelBuilder.WriteString(", ")
			}
			if r != nil {
				qualifier := types.RelativeTo(analysisInfo.TargetPackage.Types) // Adjust qualifier if needed
				sigLabelBuilder.WriteString(types.TypeString(r.Type(), qualifier))
			}
		}
		if resultsTuple.Len() > 1 {
			sigLabelBuilder.WriteString(")")
		}
	}

	sigInfo := SignatureInformation{
		Label:      sigLabelBuilder.String(),
		Parameters: parameters,
		// Documentation: // TODO: Add function documentation from analysisInfo.IdentifierDefNode if available
	}

	// Determine active parameter
	activeParam := uint32(analysisInfo.CallArgIndex)
	var activeParamPtr *uint32 // Use pointer to allow nil

	// Clamp active parameter index if it's out of bounds
	if paramsTuple != nil && activeParam < uint32(paramsTuple.Len()) {
		activeParamPtr = &activeParam // Set pointer only if valid index
	} else if sig.Variadic() && paramsTuple != nil && activeParam >= uint32(paramsTuple.Len()-1) {
		// If variadic and cursor is at or after the variadic param, activate the last param
		lastParamIndex := uint32(paramsTuple.Len() - 1)
		activeParamPtr = &lastParamIndex
	} else {
		// Index is out of bounds and not variadic case, leave activeParameter as nil
		sigLogger.Debug("Active parameter index out of bounds", "calculated_index", analysisInfo.CallArgIndex, "num_params", paramsTuple.Len())
	}

	sigHelp := &SignatureHelp{
		Signatures:      []SignatureInformation{sigInfo},
		ActiveSignature: new(uint32), // Pointer to 0 (index of the first signature)
		ActiveParameter: activeParamPtr,
	}

	sigLogger.Info("Signature help provided", "active_param_index", activeParamPtr)
	return sigHelp, nil
}
