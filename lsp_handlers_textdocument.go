// deepcomplete/lsp_handlers_textdocument.go
// Contains LSP method handlers related to text document synchronization and language features
// (didOpen, didChange, didClose, completion, hover, definition).
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
// It adds the opened file to the server's state and triggers diagnostics.
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

	// Validate URI before triggering diagnostics, pass logger
	absPath, pathErr := ValidateAndGetFilePath(string(uri), openLogger) // Util func
	if pathErr != nil {
		openLogger.Error("Invalid URI in didOpen, cannot trigger diagnostics", "error", pathErr)
		s.sendShowMessage(MessageTypeError, fmt.Sprintf("Invalid document URI: %v", pathErr))
		return nil, nil
	}

	// Trigger diagnostics in background, pass the specific logger
	go s.triggerDiagnostics(uri, version, content, absPath, openLogger)
	return nil, nil
}

// handleDidChange handles the 'textDocument/didChange' notification.
// It updates the file content in the server's state (only Full sync supported) and triggers diagnostics.
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

	// Validate URI before updating cache or triggering diagnostics, pass logger
	absPath, pathErr := ValidateAndGetFilePath(string(uri), changeLogger) // Util func
	if pathErr != nil {
		changeLogger.Error("Invalid URI in didChange", "error", pathErr)
		s.sendShowMessage(MessageTypeError, fmt.Sprintf("Invalid document URI: %v", pathErr))
		return nil, nil
	}

	s.filesMu.Lock()
	currentFile, exists := s.files[uri]
	if !exists || version > currentFile.Version {
		s.files[uri] = &OpenFile{
			URI:     uri,
			Content: newContent,
			Version: version,
		}
		changeLogger.Debug("Updated file cache")
		if s.completer.analyzer != nil {
			dir := filepath.Dir(absPath)
			if err := s.completer.InvalidateMemoryCacheForURI(string(uri), version); err != nil {
				changeLogger.Warn("Failed to invalidate memory cache on didChange", "error", err)
			}
			if err := s.completer.InvalidateAnalyzerCache(dir); err != nil {
				changeLogger.Warn("Failed to invalidate disk cache on didChange", "dir", dir, "error", err)
			}
		}
	} else {
		changeLogger.Warn("Ignoring out-of-order didChange notification", "received_version", version, "current_version", currentFile.Version)
	}
	s.filesMu.Unlock()

	// Trigger diagnostics even if the update was ignored, pass logger
	go s.triggerDiagnostics(uri, version, newContent, absPath, changeLogger)
	return nil, nil
}

// handleDidClose handles the 'textDocument/didClose' notification.
// It removes the file from the server's state and clears its diagnostics.
func (s *Server) handleDidClose(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidCloseTextDocumentParams, logger *slog.Logger) (any, error) {
	uri := params.TextDocument.URI
	closeLogger := logger.With("uri", uri)
	closeLogger.Info("Handling textDocument/didClose")

	s.filesMu.Lock()
	delete(s.files, uri)
	s.filesMu.Unlock()

	s.publishDiagnostics(uri, nil, []LspDiagnostic{}, closeLogger) // Clear diagnostics, pass logger

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
	completionLogger.Info("Handling textDocument/completion")

	s.filesMu.RLock()
	file, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		completionLogger.Warn("Completion request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	// Convert LSP position to Go position, pass logger
	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos, completionLogger) // Util func
	if posErr != nil {
		completionLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
	}
	completionLogger = completionLogger.With("go_line", line, "go_col", col)

	// Validate URI and get absolute path, pass logger
	absPath, pathErr := ValidateAndGetFilePath(string(uri), completionLogger) // Util func
	if pathErr != nil {
		completionLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	// --- Get Completion Text ---
	completionCtx, cancelCompletion := context.WithTimeout(ctx, 10*time.Second) // Timeout for LLM call
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
			// Proceed without specific kind, but log the analysis failure
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
	completionKind := CompletionItemKindSnippet // Default to Snippet
	// Only run analysis if AST is enabled in config
	currentConfig := s.completer.GetCurrentConfig()
	if currentConfig.UseAst {
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 2*time.Second) // Shorter timeout for kind analysis
		defer cancelAnalysis()
		// Analyze to find identifier at cursor for kind mapping
		// Note: This re-analyzes, could be optimized if analysis info was cached from LLM call
		analysisInfo, kindAnalysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, file.Version, line, col)
		if kindAnalysisErr != nil {
			completionLogger.Warn("Analysis for completion kind failed, using default kind", "error", kindAnalysisErr)
		} else if analysisInfo != nil && analysisInfo.IdentifierObject != nil {
			completionKind = mapTypeToCompletionKind(analysisInfo.IdentifierObject) // Use refined mapping
			completionLogger.Debug("Determined completion kind from analysis", "kind", completionKind, "identifier", analysisInfo.IdentifierObject.Name())
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
		// TODO: Potentially convert completionText to a proper LSP snippet
		insertText = completionText
		completionLogger.Debug("Using Snippet format for completion item")
	} else {
		completionLogger.Debug("Using PlainText format for completion item")
	}

	// --- Create Completion Item ---
	item := CompletionItem{
		Label:            strings.Split(completionText, "\n")[0], // Use first line as label
		InsertText:       insertText,
		InsertTextFormat: insertTextFormat,
		Kind:             completionKind,            // Use determined or default kind
		Detail:           "DeepComplete Suggestion", // Basic detail
		// TODO: Add Documentation or more Detail based on analysis if available
	}

	return CompletionList{
		IsIncomplete: false, // Mark as complete
		Items:        []CompletionItem{item},
	}, nil
}

// handleHover generates hover information.
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

	// Convert LSP position to Go position, pass logger
	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos, hoverLogger) // Util func
	if posErr != nil {
		hoverLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil
	}
	hoverLogger = hoverLogger.With("go_line", line, "go_col", col)

	// Validate URI and get absolute path, pass logger
	absPath, pathErr := ValidateAndGetFilePath(string(uri), hoverLogger) // Util func
	if pathErr != nil {
		hoverLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	// Analyze uses the logger configured in the completer instance
	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, file.Version, line, col)

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
		hoverLogger.Warn("Analysis for hover encountered errors", "error", analysisErr)
		if analysisInfo == nil {
			return nil, nil
		}
	}
	if analysisInfo == nil {
		hoverLogger.Warn("Analysis returned nil info")
		return nil, nil
	}

	if analysisInfo.IdentifierAtCursor == nil || analysisInfo.IdentifierObject == nil {
		hoverLogger.Debug("No identifier found at cursor position for hover")
		return nil, nil
	}

	// Format hover content, passing the request-specific logger
	hoverContent := formatObjectForHover(analysisInfo.IdentifierObject, analysisInfo, hoverLogger) // From helpers_hover.go
	if hoverContent == "" {
		hoverLogger.Debug("No hover content generated for identifier", "identifier", analysisInfo.IdentifierObject.Name())
		return nil, nil
	}

	var hoverRange *LSPRange
	if analysisInfo.TargetFileSet != nil {
		// Pass request-specific logger
		lspRange, rangeErr := nodeRangeToLSPRange(analysisInfo.TargetFileSet, analysisInfo.IdentifierAtCursor, file.Content, hoverLogger) // From lsp_protocol.go
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

	hoverLogger.Info("Hover information generated successfully", "identifier", analysisInfo.IdentifierObject.Name(), "markup", markupKind)
	return HoverResult{
		Contents: MarkupContent{Kind: markupKind, Value: hoverContent},
		Range:    hoverRange,
	}, nil
}

// handleDefinition finds the definition location of the symbol under the cursor.
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

	// Convert LSP position to Go position, pass logger
	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos, defLogger) // Util func
	if posErr != nil {
		defLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil
	}
	defLogger = defLogger.With("go_line", line, "go_col", col)

	// Validate URI and get absolute path, pass logger
	absPath, pathErr := ValidateAndGetFilePath(string(uri), defLogger) // Util func
	if pathErr != nil {
		defLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	// Analyze uses the logger configured in the completer instance
	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, file.Version, line, col)

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
		defLogger.Warn("Analysis for definition encountered errors", "error", analysisErr)
		if analysisInfo == nil {
			return nil, nil
		}
	}
	if analysisInfo == nil {
		defLogger.Warn("Analysis returned nil info")
		return nil, nil
	}

	if analysisInfo.IdentifierAtCursor == nil || analysisInfo.IdentifierObject == nil {
		defLogger.Debug("No identifier found at cursor position for definition")
		return nil, nil
	}

	obj := analysisInfo.IdentifierObject
	defPos := obj.Pos()

	if !defPos.IsValid() {
		defLogger.Debug("Identifier object has invalid definition position", "identifier", obj.Name())
		return nil, nil
	}

	if analysisInfo.TargetFileSet == nil {
		defLogger.Error("TargetFileSet is nil in analysis info, cannot get definition file")
		return nil, nil
	}

	defFile := analysisInfo.TargetFileSet.File(defPos)
	if defFile == nil {
		defLogger.Error("Could not find token.File for definition position", "identifier", obj.Name(), "pos", defPos)
		return nil, nil
	}

	defFileContent, readErr := os.ReadFile(defFile.Name())
	if readErr != nil {
		defLogger.Error("Failed to read definition file content", "path", defFile.Name(), "error", readErr)
		s.sendShowMessage(MessageTypeWarning, fmt.Sprintf("Could not read definition file: %s", defFile.Name()))
		return nil, nil
	}

	// Convert the definition position to an LSP Location, passing the request-specific logger
	location, locErr := tokenPosToLSPLocation(defFile, defPos, defFileContent, defLogger) // From lsp_protocol.go
	if locErr != nil {
		defLogger.Error("Failed to convert definition position to LSP Location", "identifier", obj.Name(), "error", locErr)
		return nil, nil
	}

	defLogger.Info("Definition found", "identifier", obj.Name(), "location_uri", location.URI, "location_line", location.Range.Start.Line)
	return []Location{*location}, nil
}
