// deepcomplete/lsp_server.go
// Implements the Language Server Protocol (LSP) server logic.
package deepcomplete

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"expvar" // For metrics publishing
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug" // For panic recovery
	"strings"
	"sync"
	"time"

	"github.com/sourcegraph/jsonrpc2"
)

// ============================================================================
// LSP Server Implementation
// ============================================================================

// Server represents the LSP server instance.
type Server struct {
	conn           *jsonrpc2.Conn
	logger         *slog.Logger
	completer      *DeepCompleter // The core completion service
	files          map[DocumentURI]*OpenFile
	filesMu        sync.RWMutex
	config         Config // Current effective configuration
	clientCaps     ClientCapabilities
	serverInfo     *ServerInfo
	initParams     *InitializeParams
	requestTracker *RequestTracker // Cycle 5: Request tracker
}

// OpenFile represents a file currently open in the client editor.
type OpenFile struct {
	URI     DocumentURI
	Content []byte
	Version int
}

// NewServer creates a new LSP server instance.
func NewServer(completer *DeepCompleter, logger *slog.Logger, version string) *Server {
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil)) // Default to discarding logs if none provided
	}
	s := &Server{
		logger:    logger,
		completer: completer,
		files:     make(map[DocumentURI]*OpenFile),
		config:    completer.GetCurrentConfig(), // Get initial config
		serverInfo: &ServerInfo{
			Name:    "DeepComplete LSP",
			Version: version, // Use provided version
		},
		requestTracker: NewRequestTracker(), // Cycle 5: Initialize tracker
	}
	publishExpvarMetrics(s) // Publish metrics
	return s
}

// Run starts the LSP server, listening on stdin/stdout.
func (s *Server) Run(r io.Reader, w io.Writer) {
	s.logger.Info("Starting LSP server run loop")

	stream := &stdrwc{r: r, w: w}
	objectStream := jsonrpc2.NewPlainObjectStream(stream) // Wrap stream
	handler := jsonrpc2.HandlerWithError(s.handle)

	s.conn = jsonrpc2.NewConn(context.Background(), objectStream, handler) // Use objectStream
	s.logger.Info("JSON-RPC connection established")

	<-s.conn.DisconnectNotify() // Block until connection closes
	s.logger.Info("JSON-RPC connection closed")
}

// stdrwc is a simple ReadWriteCloser that wraps stdin/stdout without closing them.
type stdrwc struct {
	r io.Reader
	w io.Writer
}

func (s *stdrwc) Read(p []byte) (int, error)  { return s.r.Read(p) }
func (s *stdrwc) Write(p []byte) (int, error) { return s.w.Write(p) }
func (s *stdrwc) Close() error                { return nil } // Do nothing

// handle routes incoming LSP requests/notifications to appropriate methods.
func (s *Server) handle(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request) (result any, err error) {
	methodLogger := s.logger.With("method", req.Method, "is_notification", req.Notif)
	isRequest := req.ID != (jsonrpc2.ID{})
	if isRequest {
		methodLogger = methodLogger.With("req_id", req.ID)
	}
	methodLogger.Debug("Received request/notification")

	// Panic recovery
	defer func() {
		if r := recover(); r != nil {
			stack := string(debug.Stack())
			methodLogger.Error("Panic recovered in handler", "panic_value", r, "stack", stack)

			// Explicitly marshal the panic message string into json.RawMessage for the Data field.
			panicMsg := fmt.Sprintf("Panic: %v", r)
			panicData, marshalErr := json.Marshal(panicMsg)
			if marshalErr != nil {
				methodLogger.Error("Failed to marshal panic message for error data", "error", marshalErr)
				// Fallback if marshaling fails
				panicData = json.RawMessage(`"failed to marshal panic data"`)
			}
			rawPanicData := json.RawMessage(panicData) // Cast to json.RawMessage

			err = &jsonrpc2.Error{
				Code:    int64(JsonRpcInternalError), // Cast code
				Message: fmt.Sprintf("Internal server error in method %s", req.Method),
				Data:    &rawPanicData, // Assign pointer to rawPanicData
			}
			result = nil
		}
	}()

	// Request Cancellation Handling
	if isRequest {
		s.requestTracker.Add(req.ID, ctx)
		defer s.requestTracker.Remove(req.ID)
	}
	select {
	case <-ctx.Done():
		methodLogger.Warn("Request context cancelled before processing started", "error", ctx.Err())
		return nil, &jsonrpc2.Error{Code: int64(JsonRpcRequestCancelled), Message: "Request cancelled"} // Cast code
	default: // Continue processing
	}

	// Helper to unmarshal params
	unmarshalParams := func(target any) error {
		if req.Params == nil {
			return errors.New("params field is null")
		}
		return json.Unmarshal(*req.Params, target) // Pass dereferenced *req.Params
	}

	switch req.Method {
	case "initialize":
		var params InitializeParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal initialize params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid initialize params: %v", err)} // Cast code
		}
		s.clientCaps = params.Capabilities
		s.initParams = &params
		return s.handleInitialize(ctx, conn, req, params)

	case "initialized":
		methodLogger.Info("Client initialized notification received")
		// Config error check removed, handled in main
		return nil, nil

	case "shutdown":
		methodLogger.Info("Shutdown request received")
		return nil, nil

	case "exit":
		methodLogger.Info("Exit notification received")
		if s.conn != nil {
			s.conn.Close()
		}
		return nil, nil

	case "textDocument/didOpen":
		var params DidOpenTextDocumentParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal didOpen params", "error", err)
			return nil, nil // Ignore notification errors
		}
		return s.handleDidOpen(ctx, conn, req, params)

	case "textDocument/didChange":
		var params DidChangeTextDocumentParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal didChange params", "error", err)
			return nil, nil // Ignore notification errors
		}
		return s.handleDidChange(ctx, conn, req, params)

	case "textDocument/didClose":
		var params DidCloseTextDocumentParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal didClose params", "error", err)
			return nil, nil // Ignore notification errors
		}
		return s.handleDidClose(ctx, conn, req, params)

	case "textDocument/completion":
		var params CompletionParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal completion params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid completion params: %v", err)} // Cast code
		}
		return s.handleCompletion(ctx, conn, req, params)

	case "textDocument/hover":
		var params HoverParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal hover params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid hover params: %v", err)} // Cast code
		}
		return s.handleHover(ctx, conn, req, params)

	case "textDocument/definition":
		var params DefinitionParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal definition params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid definition params: %v", err)} // Cast code
		}
		return s.handleDefinition(ctx, conn, req, params)

	case "workspace/didChangeConfiguration":
		var params DidChangeConfigurationParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal didChangeConfiguration params", "error", err)
			return nil, nil // Ignore notification errors
		}
		return s.handleDidChangeConfiguration(ctx, conn, req, params)

	case "$/cancelRequest":
		var params CancelParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal cancelRequest params", "error", err)
			return nil, nil // Ignore notification errors
		}
		// Reconstruct jsonrpc2.ID from params.ID (any)
		var cancelID jsonrpc2.ID
		switch idVal := params.ID.(type) {
		case float64: // JSON numbers are often float64
			numVal := uint64(idVal) // Use uint64 as confirmed by library definition
			cancelID = jsonrpc2.ID{Num: numVal}
		case string:
			cancelID = jsonrpc2.ID{Str: idVal, IsString: true} // Set IsString flag
		default:
			methodLogger.Warn("Could not determine type of cancel request ID", "id_value", params.ID, "id_type", fmt.Sprintf("%T", params.ID))
			return nil, nil // Ignore if ID type is unexpected
		}

		s.requestTracker.Cancel(cancelID) // Pass reconstructed ID
		methodLogger.Info("Cancellation request processed", "cancelled_id", cancelID)
		return nil, nil

	default:
		methodLogger.Warn("Unhandled LSP method")
		return nil, &jsonrpc2.Error{Code: int64(JsonRpcMethodNotFound), Message: fmt.Sprintf("Method not supported: %s", req.Method)} // Cast code
	}
}

// ============================================================================
// LSP Method Handlers
// ============================================================================

func (s *Server) handleInitialize(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params InitializeParams) (any, error) {
	s.logger.Info("Handling initialize request", "client_name", params.ClientInfo.Name, "client_version", params.ClientInfo.Version)

	serverCapabilities := ServerCapabilities{
		TextDocumentSync: &TextDocumentSyncOptions{
			OpenClose: true,
			Change:    TextDocumentSyncKindFull,
		},
		CompletionProvider: &CompletionOptions{},
		HoverProvider:      true,
		DefinitionProvider: true,
	}

	result := InitializeResult{
		Capabilities: serverCapabilities,
		ServerInfo:   s.serverInfo,
	}

	s.logger.Info("Initialization successful", "server_capabilities", result.Capabilities)
	return result, nil
}

func (s *Server) handleDidOpen(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidOpenTextDocumentParams) (any, error) {
	uri := params.TextDocument.URI
	version := params.TextDocument.Version
	content := []byte(params.TextDocument.Text)
	s.logger.Info("Handling textDocument/didOpen", "uri", uri, "version", version, "size", len(content))

	s.filesMu.Lock()
	s.files[uri] = &OpenFile{
		URI:     uri,
		Content: content,
		Version: version,
	}
	s.filesMu.Unlock()

	// Validate URI before triggering diagnostics
	absPath, pathErr := ValidateAndGetFilePath(string(uri))
	if pathErr != nil {
		s.logger.Error("Invalid URI in didOpen, cannot trigger diagnostics", "uri", uri, "error", pathErr)
		s.sendShowMessage(MessageTypeError, fmt.Sprintf("Invalid document URI: %v", pathErr))
		return nil, nil // Don't return error for notification
	}

	go s.triggerDiagnostics(uri, version, content, absPath)
	return nil, nil
}

func (s *Server) handleDidChange(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidChangeTextDocumentParams) (any, error) {
	uri := params.TextDocument.URI
	version := params.TextDocument.Version
	if len(params.ContentChanges) == 0 {
		s.logger.Warn("Received didChange notification with no content changes", "uri", uri, "version", version)
		return nil, nil
	}
	// For Full sync, the last change contains the full document content
	newContent := []byte(params.ContentChanges[len(params.ContentChanges)-1].Text)
	s.logger.Info("Handling textDocument/didChange", "uri", uri, "new_version", version, "new_size", len(newContent))

	// Validate URI before updating cache or triggering diagnostics
	absPath, pathErr := ValidateAndGetFilePath(string(uri))
	if pathErr != nil {
		s.logger.Error("Invalid URI in didChange", "uri", uri, "error", pathErr)
		s.sendShowMessage(MessageTypeError, fmt.Sprintf("Invalid document URI: %v", pathErr))
		return nil, nil // Don't return error for notification
	}

	s.filesMu.Lock()
	currentFile, exists := s.files[uri]
	// Update only if the new version is higher than the stored version
	if !exists || version > currentFile.Version {
		s.files[uri] = &OpenFile{
			URI:     uri,
			Content: newContent,
			Version: version,
		}
		s.logger.Debug("Updated file cache", "uri", uri, "version", version)
		if s.completer.analyzer != nil {
			// Invalidate using the validated absolute path's directory
			dir := filepath.Dir(absPath) // Calculate dir here

			// Simple memory cache invalidation for the specific URI:
			if err := s.completer.InvalidateMemoryCacheForURI(string(uri), version); err != nil {
				s.logger.Warn("Failed to invalidate memory cache on didChange", "uri", uri, "error", err)
			}
			// Consider invalidating disk cache if go.mod changes, though that's harder to detect here.
			if err := s.completer.InvalidateAnalyzerCache(dir); err != nil {
				s.logger.Warn("Failed to invalidate disk cache on didChange", "dir", dir, "error", err)
			}
		}
	} else {
		s.logger.Warn("Ignoring out-of-order didChange notification", "uri", uri, "received_version", version, "current_version", currentFile.Version)
	}
	s.filesMu.Unlock()

	// Trigger diagnostics even if the update was ignored (client state might need update)
	go s.triggerDiagnostics(uri, version, newContent, absPath)
	return nil, nil
}

func (s *Server) handleDidClose(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidCloseTextDocumentParams) (any, error) {
	uri := params.TextDocument.URI
	s.logger.Info("Handling textDocument/didClose", "uri", uri)

	s.filesMu.Lock()
	delete(s.files, uri)
	s.filesMu.Unlock()

	s.publishDiagnostics(uri, nil, []LspDiagnostic{}) // Clear diagnostics

	if s.completer.analyzer != nil {
		if err := s.completer.InvalidateMemoryCacheForURI(string(uri), 0); err != nil {
			s.logger.Warn("Failed to invalidate memory cache on didClose", "uri", uri, "error", err)
		}
	}
	return nil, nil
}

// handleCompletion generates code completions based on the current document state and cursor position.
// ** MODIFIED: Cycle 1 - Default to Snippet kind/format for LLM completions **
func (s *Server) handleCompletion(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params CompletionParams) (any, error) {
	uri := params.TextDocument.URI
	lspPos := params.Position
	completionLogger := s.logger.With("uri", uri, "lsp_line", lspPos.Line, "lsp_char", lspPos.Character)
	completionLogger.Info("Handling textDocument/completion")

	s.filesMu.RLock()
	file, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		completionLogger.Warn("Completion request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	// Convert LSP position (0-based, UTF-16) to Go position (1-based line/col, byte offset)
	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos)
	if posErr != nil {
		completionLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		// Return empty list on position error, don't fail the request
		return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
	}
	completionLogger = completionLogger.With("go_line", line, "go_col", col)

	// Validate URI and get absolute path
	absPath, pathErr := ValidateAndGetFilePath(string(uri))
	if pathErr != nil {
		completionLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	completionCtx, cancel := context.WithTimeout(ctx, 10*time.Second) // Timeout for the entire completion operation
	defer cancel()

	var completionBuffer bytes.Buffer
	// Get completion stream from the core service
	completionErr := s.completer.GetCompletionStreamFromFile(completionCtx, absPath, file.Version, line, col, &completionBuffer)

	if completionErr != nil {
		if errors.Is(completionErr, context.DeadlineExceeded) {
			completionLogger.Warn("Completion request timed out")
			return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil // Return empty list on timeout
		}
		if errors.Is(completionErr, context.Canceled) {
			completionLogger.Info("Completion request cancelled")
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcRequestCancelled), Message: "Completion request cancelled"} // Return cancellation error
		}
		// Handle specific errors like Ollama being unavailable or analysis failing
		if errors.Is(completionErr, ErrOllamaUnavailable) {
			completionLogger.Error("Ollama unavailable", "error", completionErr)
			s.sendShowMessage(MessageTypeError, fmt.Sprintf("Completion backend error: %v", completionErr))
			return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil // Return empty list
		}
		if errors.Is(completionErr, ErrAnalysisFailed) {
			completionLogger.Warn("Code analysis failed during completion", "error", completionErr)
			// Still return empty list, don't error out the request
			return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
		}
		// Handle other unexpected errors
		completionLogger.Error("Failed to get completion stream", "error", completionErr)
		return nil, fmt.Errorf("completion failed: %w", completionErr) // Return generic error for LSP
	}

	completionText := strings.TrimSpace(completionBuffer.String())
	if completionText == "" {
		completionLogger.Info("Completion successful but result is empty")
		return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
	}

	completionLogger.Info("Completion successful", "completion_length", len(completionText))

	// Determine insert text format based on client capabilities
	insertTextFormat := PlainTextFormat // Default to plain text
	insertText := completionText
	if s.clientCaps.TextDocument != nil &&
		s.clientCaps.TextDocument.Completion != nil &&
		s.clientCaps.TextDocument.Completion.CompletionItem != nil &&
		s.clientCaps.TextDocument.Completion.CompletionItem.SnippetSupport {
		insertTextFormat = SnippetFormat
		// Use raw text as snippet for now. Could be enhanced to generate LSP snippets.
		insertText = completionText
		completionLogger.Debug("Using Snippet format for completion item")
	} else {
		completionLogger.Debug("Using PlainText format for completion item (client snippet support unknown/disabled)")
	}

	// Create the completion item
	item := CompletionItem{
		Label:            strings.Split(completionText, "\n")[0], // Use first line as label
		InsertText:       insertText,
		InsertTextFormat: insertTextFormat,
		Kind:             CompletionItemKindSnippet, // Default to Snippet kind for LLM completions
		Detail:           "DeepComplete Suggestion",
		// Consider adding documentation or more detail later
	}

	return CompletionList{
		IsIncomplete: false, // Assuming the LLM provides the full completion in one go
		Items:        []CompletionItem{item},
	}, nil
}

func (s *Server) handleHover(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params HoverParams) (any, error) {
	uri := params.TextDocument.URI
	lspPos := params.Position
	hoverLogger := s.logger.With("uri", uri, "lsp_line", lspPos.Line, "lsp_char", lspPos.Character)
	hoverLogger.Info("Handling textDocument/hover")

	s.filesMu.RLock()
	file, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		hoverLogger.Warn("Hover request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos)
	if posErr != nil {
		hoverLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil // Return nil result for hover on position error
	}
	hoverLogger = hoverLogger.With("go_line", line, "go_col", col)

	// Validate URI and get absolute path
	absPath, pathErr := ValidateAndGetFilePath(string(uri))
	if pathErr != nil {
		hoverLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 15*time.Second) // Timeout for analysis
	defer cancel()

	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, file.Version, line, col)
	if analysisErr != nil {
		hoverLogger.Warn("Analysis for hover encountered errors", "error", analysisErr)
		// Proceed even with non-fatal errors if info is available
		if analysisInfo == nil {
			return nil, nil // Return nil result if analysis failed completely
		}
	}
	if analysisInfo == nil {
		hoverLogger.Warn("Analysis returned nil info")
		return nil, nil // Return nil result
	}

	// Check if an identifier was found at the cursor and its object resolved
	if analysisInfo.IdentifierAtCursor == nil || analysisInfo.IdentifierObject == nil {
		hoverLogger.Debug("No identifier found at cursor position for hover")
		return nil, nil // Return nil result
	}

	// Format the hover content using the resolved object and analysis info
	hoverContent := formatObjectForHover(analysisInfo.IdentifierObject, analysisInfo, hoverLogger)
	if hoverContent == "" {
		hoverLogger.Debug("No hover content generated for identifier", "identifier", analysisInfo.IdentifierObject.Name())
		return nil, nil // Return nil result
	}

	// Determine the range of the identifier for highlighting in the editor
	var hoverRange *LSPRange
	if analysisInfo.TargetFileSet != nil {
		// Use the identifier node itself for the hover range
		lspRange, rangeErr := nodeRangeToLSPRange(analysisInfo.TargetFileSet, analysisInfo.IdentifierAtCursor, file.Content, hoverLogger)
		if rangeErr == nil {
			hoverRange = lspRange
		} else {
			hoverLogger.Warn("Could not determine range for hover identifier", "error", rangeErr)
		}
	}

	// Determine the best markup kind based on client capabilities
	markupKind := MarkupKindPlainText // Default to plain text
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
// ** MODIFIED: Cycle 1 - Pass file content to tokenPosToLSPLocation **
func (s *Server) handleDefinition(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DefinitionParams) (any, error) {
	uri := params.TextDocument.URI
	lspPos := params.Position
	defLogger := s.logger.With("uri", uri, "lsp_line", lspPos.Line, "lsp_char", lspPos.Character)
	defLogger.Info("Handling textDocument/definition")

	s.filesMu.RLock()
	file, ok := s.files[uri]
	s.filesMu.RUnlock()

	if !ok {
		defLogger.Warn("Definition request for unknown file")
		return nil, fmt.Errorf("document not open: %s", uri)
	}

	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos)
	if posErr != nil {
		defLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil // Return nil result on position error
	}
	defLogger = defLogger.With("go_line", line, "go_col", col)

	// Validate URI and get absolute path
	absPath, pathErr := ValidateAndGetFilePath(string(uri))
	if pathErr != nil {
		defLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 15*time.Second) // Timeout for analysis
	defer cancel()

	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, file.Version, line, col)
	if analysisErr != nil {
		defLogger.Warn("Analysis for definition encountered errors", "error", analysisErr)
		if analysisInfo == nil {
			return nil, nil // Return nil result if analysis failed completely
		}
	}
	if analysisInfo == nil {
		defLogger.Warn("Analysis returned nil info")
		return nil, nil // Return nil result
	}

	// Check if an identifier was found and resolved
	if analysisInfo.IdentifierAtCursor == nil || analysisInfo.IdentifierObject == nil {
		defLogger.Debug("No identifier found at cursor position for definition")
		return nil, nil // Return nil result
	}

	obj := analysisInfo.IdentifierObject
	defPos := obj.Pos() // Get the definition position from the types.Object

	if !defPos.IsValid() {
		defLogger.Debug("Identifier object has invalid definition position", "identifier", obj.Name())
		return nil, nil // Return nil result
	}

	if analysisInfo.TargetFileSet == nil {
		defLogger.Error("TargetFileSet is nil in analysis info, cannot get definition file")
		return nil, nil // Return nil result
	}

	// Find the token.File corresponding to the definition position
	defFile := analysisInfo.TargetFileSet.File(defPos)
	if defFile == nil {
		defLogger.Error("Could not find token.File for definition position", "identifier", obj.Name(), "pos", defPos)
		return nil, nil // Return nil result
	}

	// Read definition file content (might be different from the current file)
	// This is needed for accurate LSP position conversion
	defFileContent, readErr := os.ReadFile(defFile.Name())
	if readErr != nil {
		defLogger.Error("Failed to read definition file content", "path", defFile.Name(), "error", readErr)
		s.sendShowMessage(MessageTypeWarning, fmt.Sprintf("Could not read definition file: %s", defFile.Name()))
		return nil, nil // Return nil result if we can't read the definition file
	}

	// Convert the definition position (token.Pos) to an LSP Location
	location, locErr := tokenPosToLSPLocation(defFile, defPos, defFileContent, defLogger)
	if locErr != nil {
		defLogger.Error("Failed to convert definition position to LSP Location", "identifier", obj.Name(), "error", locErr)
		return nil, nil // Return nil result
	}

	defLogger.Info("Definition found", "identifier", obj.Name(), "location_uri", location.URI, "location_line", location.Range.Start.Line)
	// Return result as a slice, even if single location, as per LSP spec
	return []Location{*location}, nil
}

func (s *Server) handleDidChangeConfiguration(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidChangeConfigurationParams) (any, error) {
	s.logger.Info("Handling workspace/didChangeConfiguration")

	// Define a struct that mirrors the expected nested structure from the client.
	// Adjust the "DeepComplete" key if your client sends it differently (e.g., "deepcomplete").
	var changedSettings struct {
		DeepComplete FileConfig `json:"deepcomplete"` // Match the key sent by the client
	}

	// Attempt to unmarshal the *entire* settings object into our struct.
	if err := json.Unmarshal(params.Settings, &changedSettings); err != nil {
		s.logger.Error("Failed to unmarshal workspace/didChangeConfiguration settings", "error", err, "raw_settings", string(params.Settings))
		// Attempt to unmarshal directly into FileConfig if nesting fails
		var directFileCfg FileConfig
		if directErr := json.Unmarshal(params.Settings, &directFileCfg); directErr == nil {
			s.logger.Info("Successfully unmarshalled settings directly into FileConfig (no 'deepcomplete' nesting)")
			changedSettings.DeepComplete = directFileCfg // Use the directly unmarshalled config
		} else {
			s.logger.Error("Also failed to unmarshal settings directly into FileConfig", "direct_error", directErr)
			return nil, nil // Ignore notification errors
		}
	}

	newConfig := s.completer.GetCurrentConfig() // Get current config as base
	fileCfg := changedSettings.DeepComplete     // Use the potentially nested or direct config
	mergedFields := 0

	// Merge fields only if they were present in the received settings (non-nil pointers)
	if fileCfg.OllamaURL != nil {
		newConfig.OllamaURL = *fileCfg.OllamaURL
		mergedFields++
	}
	if fileCfg.Model != nil {
		newConfig.Model = *fileCfg.Model
		mergedFields++
	}
	if fileCfg.MaxTokens != nil {
		newConfig.MaxTokens = *fileCfg.MaxTokens
		mergedFields++
	}
	if fileCfg.Stop != nil {
		newConfig.Stop = *fileCfg.Stop
		mergedFields++
	}
	if fileCfg.Temperature != nil {
		newConfig.Temperature = *fileCfg.Temperature
		mergedFields++
	}
	if fileCfg.LogLevel != nil {
		newConfig.LogLevel = *fileCfg.LogLevel
		mergedFields++
		// Log level change intention, but rely on UpdateConfig validation
		s.logger.Info("Log level configuration change received", "new_level_setting", newConfig.LogLevel)
	}
	if fileCfg.UseAst != nil {
		newConfig.UseAst = *fileCfg.UseAst
		mergedFields++
	}
	if fileCfg.UseFim != nil {
		newConfig.UseFim = *fileCfg.UseFim
		mergedFields++
	}
	if fileCfg.MaxPreambleLen != nil {
		newConfig.MaxPreambleLen = *fileCfg.MaxPreambleLen
		mergedFields++
	}
	if fileCfg.MaxSnippetLen != nil {
		newConfig.MaxSnippetLen = *fileCfg.MaxSnippetLen
		mergedFields++
	}

	if mergedFields > 0 {
		s.logger.Info("Applying configuration changes from client", "fields_merged", mergedFields)
		// UpdateConfig performs validation internally
		if err := s.completer.UpdateConfig(newConfig); err != nil {
			s.logger.Error("Failed to apply updated configuration", "error", err)
			s.sendShowMessage(MessageTypeError, fmt.Sprintf("Failed to apply configuration update: %v", err))
		} else {
			// Update the server's local copy after successful update in completer
			s.config = s.completer.GetCurrentConfig()
			s.logger.Info("Server configuration updated successfully via workspace/didChangeConfiguration")
			// Potentially update server's logger level if it changed
			newLevel, parseErr := ParseLogLevel(s.config.LogLevel)
			if parseErr == nil {
				// Assuming logger supports dynamic level changes or recreation
				// This part depends heavily on how the logger is implemented/passed
				s.logger.Info("Attempting to update logger level (implementation specific)", "new_level", newLevel)
				// Example: Recreate the logger (less ideal if passed around)
				// logWriter := io.MultiWriter(os.Stderr, logFile) // Assuming logFile is accessible
				// handlerOpts := slog.HandlerOptions{Level: newLevel, AddSource: true}
				// handler := slog.NewTextHandler(logWriter, &handlerOpts)
				// s.logger = slog.New(handler)
				// slog.SetDefault(s.logger) // Update default logger if used globally
			} else {
				s.logger.Warn("Cannot update logger level due to parse error", "level_string", s.config.LogLevel, "error", parseErr)
			}
		}
	} else {
		s.logger.Debug("No relevant configuration changes found in workspace/didChangeConfiguration notification")
	}

	return nil, nil
}

// ============================================================================
// LSP Notification Sending Helpers
// ============================================================================

func (s *Server) sendShowMessage(msgType MessageType, message string) {
	if s.conn == nil {
		s.logger.Warn("Cannot send showMessage: connection is nil")
		return
	}
	params := ShowMessageParams{Type: msgType, Message: message}
	ctx := context.Background() // Use background context for notifications
	if err := s.conn.Notify(ctx, "window/showMessage", params); err != nil {
		s.logger.Error("Failed to send window/showMessage notification", "error", err, "message_type", msgType)
	} else {
		s.logger.Debug("Sent window/showMessage notification", "message_type", msgType)
	}
}

func (s *Server) publishDiagnostics(uri DocumentURI, version *int, diagnostics []LspDiagnostic) {
	if s.conn == nil {
		s.logger.Warn("Cannot publish diagnostics: connection is nil", "uri", uri)
		return
	}
	params := PublishDiagnosticsParams{
		URI:         uri,
		Version:     version, // Pass version if available
		Diagnostics: diagnostics,
	}
	ctx := context.Background() // Use background context for notifications
	if err := s.conn.Notify(ctx, "textDocument/publishDiagnostics", params); err != nil {
		s.logger.Error("Failed to send textDocument/publishDiagnostics notification", "error", err, "uri", uri, "diagnostic_count", len(diagnostics))
	} else {
		s.logger.Info("Published diagnostics", "uri", uri, "diagnostic_count", len(diagnostics), "version", version)
	}
}

// triggerDiagnostics performs analysis and publishes diagnostics. Requires absPath.
// ** MODIFIED: Cycle 1 - Improved range conversion and error handling **
func (s *Server) triggerDiagnostics(uri DocumentURI, version int, content []byte, absPath string) {
	diagLogger := s.logger.With("uri", uri, "version", version, "absPath", absPath, "operation", "triggerDiagnostics")
	diagLogger.Info("Triggering background analysis for diagnostics")

	// Path validation already done by caller (didOpen/didChange)

	analysisCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Analyze using the absolute path, use placeholder line/col (1,1) as diagnostics cover the whole file
	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, version, 1, 1)

	lspDiagnostics := []LspDiagnostic{}
	if analysisInfo != nil && len(analysisInfo.Diagnostics) > 0 {
		diagLogger.Debug("Converting internal diagnostics to LSP format", "count", len(analysisInfo.Diagnostics))
		for _, diag := range analysisInfo.Diagnostics {
			// Convert internal range (byte offsets) to LSP range (UTF-16)
			lspRange, err := internalRangeToLSPRange(content, diag.Range, diagLogger)
			if err != nil {
				diagLogger.Warn("Failed to convert diagnostic range, skipping diagnostic", "internal_range", diag.Range, "error", err, "message", diag.Message)
				continue // Skip diagnostics where range conversion fails
			}
			lspDiagnostics = append(lspDiagnostics, LspDiagnostic{
				Range:    *lspRange,
				Severity: mapInternalSeverityToLSP(diag.Severity),
				Code:     diag.Code,
				Source:   diag.Source,
				Message:  diag.Message,
			})
		}
	} else if analysisInfo == nil {
		diagLogger.Warn("Analysis for diagnostics returned nil info", "analysis_error", analysisErr)
	} else {
		diagLogger.Debug("No internal diagnostics found during analysis")
	}

	// Log analysis errors
	if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
		// Log fatal analysis errors more prominently
		diagLogger.Error("Fatal error during diagnostic analysis", "error", analysisErr)
		// Optionally, send a single diagnostic indicating a server-side analysis failure
		// lspDiagnostics = append(lspDiagnostics, createServerAnalysisErrorDiagnostic(analysisErr))
	} else if analysisErr != nil {
		// Log non-fatal errors as warnings
		diagLogger.Warn("Analysis for diagnostics completed with non-fatal errors", "error", analysisErr)
	}

	s.publishDiagnostics(uri, &version, lspDiagnostics)
}

// internalRangeToLSPRange converts internal byte-offset range to LSP UTF-16 range.
// ** MODIFIED: Cycle 1 - Improved error handling and validation **
func internalRangeToLSPRange(content []byte, internalRange Range, logger *slog.Logger) (*LSPRange, error) {
	if content == nil {
		return nil, errors.New("cannot convert range: content is nil")
	}
	// Validate internal range offsets
	if internalRange.Start.Character < 0 || internalRange.End.Character < internalRange.Start.Character || internalRange.End.Character > len(content) {
		return nil, fmt.Errorf("invalid internal byte offset range: start=%d, end=%d, content_len=%d", internalRange.Start.Character, internalRange.End.Character, len(content))
	}

	startLine, startChar, startErr := byteOffsetToLSPPosition(content, internalRange.Start.Character, logger)
	if startErr != nil {
		return nil, fmt.Errorf("failed converting start offset %d: %w", internalRange.Start.Character, startErr)
	}
	endLine, endChar, endErr := byteOffsetToLSPPosition(content, internalRange.End.Character, logger)
	if endErr != nil {
		return nil, fmt.Errorf("failed converting end offset %d: %w", internalRange.End.Character, endErr)
	}

	// Basic validation: ensure start is not after end in LSP coordinates
	if startLine > endLine || (startLine == endLine && startChar > endChar) {
		logger.Warn("Calculated invalid LSP range (end < start), adjusting end to start", "start_line", startLine, "start_char", startChar, "end_line", endLine, "end_char", endChar)
		// Adjust end to be same as start to create a zero-length range at the start position
		endLine = startLine
		endChar = startChar
	}

	return &LSPRange{
		Start: LSPPosition{Line: startLine, Character: startChar},
		End:   LSPPosition{Line: endLine, Character: endChar},
	}, nil
}

// mapInternalSeverityToLSP maps internal severity levels to LSP severity levels.
func mapInternalSeverityToLSP(internalSeverity DiagnosticSeverity) LspDiagnosticSeverity {
	switch internalSeverity {
	case SeverityError:
		return LspSeverityError
	case SeverityWarning:
		return LspSeverityWarning
	case SeverityInfo:
		return LspSeverityInfo
	case SeverityHint:
		return LspSeverityHint
	default:
		slog.Warn("Unknown internal diagnostic severity, defaulting to Error", "internal_severity", internalSeverity)
		return LspSeverityError
	}
}

// ============================================================================
// Metrics Publishing (Cycle 4)
// ============================================================================

func publishExpvarMetrics(s *Server) {
	startTime := time.Now()
	expvar.NewString("serverInfo.name").Set(s.serverInfo.Name)
	expvar.NewString("serverInfo.version").Set(s.serverInfo.Version)
	expvar.NewString("serverStartTime").Set(startTime.Format(time.RFC3339))
	expvar.Publish("goroutines", expvar.Func(func() any { return runtime.NumGoroutine() }))
	expvar.Publish("memory.allocBytes", expvar.Func(func() any {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return m.Alloc
	}))
	expvar.Publish("lsp.openFiles", expvar.Func(func() any {
		s.filesMu.RLock()
		defer s.filesMu.RUnlock()
		return len(s.files)
	}))
	expvar.Publish("lsp.pendingRequests", expvar.Func(func() any { return s.requestTracker.Count() }))

	// Publish cache metrics if analyzer and cache are available
	if s.completer != nil && s.completer.analyzer != nil && s.completer.analyzer.MemoryCacheEnabled() {
		expvar.Publish("cache.memory.hits", expvar.Func(func() any {
			m := s.completer.analyzer.GetMemoryCacheMetrics()
			if m != nil {
				return m.Hits()
			}
			return 0
		}))
		expvar.Publish("cache.memory.misses", expvar.Func(func() any {
			m := s.completer.analyzer.GetMemoryCacheMetrics()
			if m != nil {
				return m.Misses()
			}
			return 0
		}))
		expvar.Publish("cache.memory.costAdded", expvar.Func(func() any {
			m := s.completer.analyzer.GetMemoryCacheMetrics()
			if m != nil {
				return m.CostAdded()
			}
			return 0
		}))
		expvar.Publish("cache.memory.costEvicted", expvar.Func(func() any {
			m := s.completer.analyzer.GetMemoryCacheMetrics()
			if m != nil {
				return m.CostEvicted()
			}
			return 0
		}))
		expvar.Publish("cache.memory.keysAdded", expvar.Func(func() any {
			m := s.completer.analyzer.GetMemoryCacheMetrics()
			if m != nil {
				return m.KeysAdded()
			}
			return 0
		}))
		expvar.Publish("cache.memory.keysEvicted", expvar.Func(func() any {
			m := s.completer.analyzer.GetMemoryCacheMetrics()
			if m != nil {
				return m.KeysEvicted()
			}
			return 0
		}))
	} else {
		// Publish zero values if cache is not enabled
		expvar.Publish("cache.memory.hits", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.misses", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.costAdded", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.costEvicted", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.keysAdded", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.keysEvicted", expvar.Func(func() any { return 0 }))
	}
	s.logger.Info("Expvar metrics published")
}

// ============================================================================
// Request Cancellation Tracker (Cycle 5)
// ============================================================================

// RequestTracker manages cancellation contexts for ongoing LSP requests.
type RequestTracker struct {
	mu       sync.Mutex
	requests map[jsonrpc2.ID]context.CancelFunc // Map request ID to its cancel function
}

// NewRequestTracker creates a new tracker.
func NewRequestTracker() *RequestTracker {
	return &RequestTracker{
		requests: make(map[jsonrpc2.ID]context.CancelFunc),
	}
}

// Add registers a request ID and its associated context's cancel function.
func (rt *RequestTracker) Add(id jsonrpc2.ID, ctx context.Context) {
	// Only add if the ID is actually set (not a notification)
	if id == (jsonrpc2.ID{}) {
		return
	} // Correct check
	rt.mu.Lock()
	defer rt.mu.Unlock()
	// Create a new cancellable context derived from the request context
	reqCtx, cancel := context.WithCancel(ctx)
	rt.requests[id] = cancel
	_ = reqCtx // Avoid unused variable error if reqCtx isn't used otherwise
}

// Remove deregisters a request ID.
func (rt *RequestTracker) Remove(id jsonrpc2.ID) {
	// Only remove if the ID is actually set
	if id == (jsonrpc2.ID{}) {
		return
	} // Correct check
	rt.mu.Lock()
	defer rt.mu.Unlock()
	// Check if the cancel function exists before deleting
	if _, ok := rt.requests[id]; ok {
		delete(rt.requests, id)
	}
}

// Cancel finds the cancel function for a request ID and calls it.
func (rt *RequestTracker) Cancel(id jsonrpc2.ID) {
	// Only cancel if the ID is actually set
	if id == (jsonrpc2.ID{}) { // Correct check
		slog.Debug("Cancel request ignored for unset ID")
		return
	}
	rt.mu.Lock()
	cancel, found := rt.requests[id]
	if found {
		// Remove immediately upon finding to prevent double cancellation attempts
		delete(rt.requests, id)
	}
	rt.mu.Unlock() // Unlock before calling cancel

	if found {
		slog.Debug("Calling cancel function for request", "id", id)
		cancel() // Call cancel function outside the lock
	} else {
		slog.Debug("Cancel function not found for request ID (already finished or invalid?)", "id", id)
	}
}

// Count returns the number of currently tracked requests.
func (rt *RequestTracker) Count() int {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return len(rt.requests)
}
