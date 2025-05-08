// deepcomplete/lsp_handlers_textdocument.go
// Contains LSP method handlers related to text document synchronization and language features
// (didOpen, didChange, didClose, completion, hover, definition, codeAction).
package deepcomplete

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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
	file, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		completionLogger.Warn("Completion request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos, completionLogger)
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
	completionErr := s.completer.GetCompletionStreamFromFile(completionCtx, absPath, file.Version, line, col, &completionBuffer)

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
		identInfo, kindAnalysisErr := s.completer.analyzer.GetIdentifierInfo(analysisCtx, absPath, file.Version, line, col)
		if kindAnalysisErr != nil {
			completionLogger.Warn("Analysis for completion kind failed, using default kind", "error", kindAnalysisErr)
		} else if identInfo != nil && identInfo.Object != nil {
			completionKind = mapTypeToCompletionKind(identInfo.Object)
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
	file, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		hoverLogger.Warn("Hover request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos, hoverLogger)
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
	identInfo, analysisErr := s.completer.analyzer.GetIdentifierInfo(analysisCtx, absPath, file.Version, line, col)

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
	file, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		defLogger.Warn("Definition request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos, defLogger)
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
	identInfo, analysisErr := s.completer.analyzer.GetIdentifierInfo(analysisCtx, absPath, file.Version, line, col)

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
	defPos := obj.Pos() // token.Pos
	defFile := identInfo.FileSet.File(defPos)
	if defFile == nil {
		defLogger.Error("Could not find token.File for definition position", "identifier", obj.Name(), "pos", defPos)
		return nil, nil
	}

	// Read content of the definition file only if needed for conversion
	var defFileContent []byte
	var readErr error
	defFileName := defFile.Name()
	if defFileName != absPath { // Check if definition is in a different file
		defFileContent, readErr = os.ReadFile(defFileName)
		if readErr != nil {
			defLogger.Error("Failed to read definition file content", "path", defFileName, "error", readErr)
			s.sendShowMessage(MessageTypeWarning, fmt.Sprintf("Could not read definition file: %s", defFileName))
			return nil, nil
		}
	} else {
		// Use the content already held by the server for the current file
		defFileContent = file.Content
	}

	// Convert the definition token.Pos to an LSP Location
	location, locErr := tokenPosToLSPLocation(defFile, defPos, defFileContent, defLogger)
	if locErr != nil {
		defLogger.Error("Failed to convert definition position to LSP Location", "identifier", obj.Name(), "error", locErr)
		return nil, nil
	}

	defLogger.Info("Definition found", "identifier", obj.Name(), "location_uri", location.URI, "location_line", location.Range.Start.Line)
	return []Location{*location}, nil // Return as slice
}

// handleCodeAction handles the 'textDocument/codeAction' request.
// Currently a stub.
func (s *Server) handleCodeAction(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params CodeActionParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	actionLogger := logger.With("uri", uri, "range", params.Range, "req_id", req.ID)
	actionLogger.Info("Handling textDocument/codeAction (stub)")

	// TODO: Implement actual code action logic
	// - Analyze the range/diagnostics provided in params.Context
	// - Determine possible actions (e.g., quick fixes, refactors)
	// - Return a list of Command or CodeAction objects

	// Example: Return empty list for now
	return []any{}, nil
}
