// deepcomplete/lsp_server.go
// Implements the Language Server Protocol (LSP) server logic.
// Cycle 2: Updated utility function calls and cleaned comments.
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
	config         Config // Current effective configuration (from deepcomplete_types.go)
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
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	s := &Server{
		logger:    logger,
		completer: completer,
		files:     make(map[DocumentURI]*OpenFile),
		config:    completer.GetCurrentConfig(), // Get initial config
		serverInfo: &ServerInfo{
			Name:    "DeepComplete LSP",
			Version: version,
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
	objectStream := jsonrpc2.NewPlainObjectStream(stream)
	handler := jsonrpc2.HandlerWithError(s.handle)

	s.conn = jsonrpc2.NewConn(context.Background(), objectStream, handler)
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

			panicMsg := fmt.Sprintf("Panic: %v", r)
			panicData, marshalErr := json.Marshal(panicMsg)
			if marshalErr != nil {
				methodLogger.Error("Failed to marshal panic message for error data", "error", marshalErr)
				panicData = json.RawMessage(`"failed to marshal panic data"`)
			}
			rawPanicData := json.RawMessage(panicData)

			err = &jsonrpc2.Error{
				Code:    int64(JsonRpcInternalError),
				Message: fmt.Sprintf("Internal server error in method %s", req.Method),
				Data:    &rawPanicData,
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
		return nil, &jsonrpc2.Error{Code: int64(JsonRpcRequestCancelled), Message: "Request cancelled"}
	default: // Continue processing
	}

	// Helper to unmarshal params
	unmarshalParams := func(target any) error {
		if req.Params == nil {
			return errors.New("params field is null")
		}
		return json.Unmarshal(*req.Params, target)
	}

	switch req.Method {
	case "initialize":
		var params InitializeParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal initialize params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid initialize params: %v", err)}
		}
		s.clientCaps = params.Capabilities
		s.initParams = &params
		return s.handleInitialize(ctx, conn, req, params)

	case "initialized":
		methodLogger.Info("Client initialized notification received")
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
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid completion params: %v", err)}
		}
		return s.handleCompletion(ctx, conn, req, params)

	case "textDocument/hover":
		var params HoverParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal hover params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid hover params: %v", err)}
		}
		return s.handleHover(ctx, conn, req, params)

	case "textDocument/definition":
		var params DefinitionParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal definition params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid definition params: %v", err)}
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
		var cancelID jsonrpc2.ID
		switch idVal := params.ID.(type) {
		case float64:
			numVal := uint64(idVal)
			cancelID = jsonrpc2.ID{Num: numVal}
		case string:
			cancelID = jsonrpc2.ID{Str: idVal, IsString: true}
		default:
			methodLogger.Warn("Could not determine type of cancel request ID", "id_value", params.ID, "id_type", fmt.Sprintf("%T", params.ID))
			return nil, nil
		}

		s.requestTracker.Cancel(cancelID)
		methodLogger.Info("Cancellation request processed", "cancelled_id", cancelID)
		return nil, nil

	default:
		methodLogger.Warn("Unhandled LSP method")
		return nil, &jsonrpc2.Error{Code: int64(JsonRpcMethodNotFound), Message: fmt.Sprintf("Method not supported: %s", req.Method)}
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
	absPath, pathErr := ValidateAndGetFilePath(string(uri)) // Util func from deepcomplete_utils.go
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
	absPath, pathErr := ValidateAndGetFilePath(string(uri)) // Util func from deepcomplete_utils.go
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
			dir := filepath.Dir(absPath)

			// Invalidate memory cache for the specific URI
			if err := s.completer.InvalidateMemoryCacheForURI(string(uri), version); err != nil {
				s.logger.Warn("Failed to invalidate memory cache on didChange", "uri", uri, "error", err)
			}
			// Invalidate disk cache for the directory (might be too broad, consider refining later)
			if err := s.completer.InvalidateAnalyzerCache(dir); err != nil {
				s.logger.Warn("Failed to invalidate disk cache on didChange", "dir", dir, "error", err)
			}
		}
	} else {
		s.logger.Warn("Ignoring out-of-order didChange notification", "uri", uri, "received_version", version, "current_version", currentFile.Version)
	}
	s.filesMu.Unlock()

	// Trigger diagnostics even if the update was ignored
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

// handleCompletion generates code completions.
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

	// Convert LSP position to Go position
	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos) // Util func from deepcomplete_utils.go
	if posErr != nil {
		completionLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil // Return empty list
	}
	completionLogger = completionLogger.With("go_line", line, "go_col", col)

	// Validate URI and get absolute path
	absPath, pathErr := ValidateAndGetFilePath(string(uri)) // Util func from deepcomplete_utils.go
	if pathErr != nil {
		completionLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	completionCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	var completionBuffer bytes.Buffer
	// Get completion stream from the core service
	completionErr := s.completer.GetCompletionStreamFromFile(completionCtx, absPath, file.Version, line, col, &completionBuffer)

	if completionErr != nil {
		if errors.Is(completionErr, context.DeadlineExceeded) {
			completionLogger.Warn("Completion request timed out")
			return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
		}
		if errors.Is(completionErr, context.Canceled) {
			completionLogger.Info("Completion request cancelled")
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcRequestCancelled), Message: "Completion request cancelled"}
		}
		if errors.Is(completionErr, ErrOllamaUnavailable) {
			completionLogger.Error("Ollama unavailable", "error", completionErr)
			s.sendShowMessage(MessageTypeError, fmt.Sprintf("Completion backend error: %v", completionErr))
			return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
		}
		if errors.Is(completionErr, ErrAnalysisFailed) {
			completionLogger.Warn("Code analysis failed during completion", "error", completionErr)
			return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
		}
		completionLogger.Error("Failed to get completion stream", "error", completionErr)
		return nil, fmt.Errorf("completion failed: %w", completionErr)
	}

	completionText := strings.TrimSpace(completionBuffer.String())
	if completionText == "" {
		completionLogger.Info("Completion successful but result is empty")
		return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}, nil
	}

	completionLogger.Info("Completion successful", "completion_length", len(completionText))

	// Determine insert text format based on client capabilities
	insertTextFormat := PlainTextFormat
	insertText := completionText
	if s.clientCaps.TextDocument != nil &&
		s.clientCaps.TextDocument.Completion != nil &&
		s.clientCaps.TextDocument.Completion.CompletionItem != nil &&
		s.clientCaps.TextDocument.Completion.CompletionItem.SnippetSupport {
		insertTextFormat = SnippetFormat
		insertText = completionText // Use raw text as snippet for now
		completionLogger.Debug("Using Snippet format for completion item")
	} else {
		completionLogger.Debug("Using PlainText format for completion item")
	}

	// Create the completion item
	item := CompletionItem{
		Label:            strings.Split(completionText, "\n")[0],
		InsertText:       insertText,
		InsertTextFormat: insertTextFormat,
		Kind:             CompletionItemKindSnippet, // Default to Snippet kind
		Detail:           "DeepComplete Suggestion",
	}

	return CompletionList{
		IsIncomplete: false,
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

	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos) // Util func
	if posErr != nil {
		hoverLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil // Return nil result
	}
	hoverLogger = hoverLogger.With("go_line", line, "go_col", col)

	absPath, pathErr := ValidateAndGetFilePath(string(uri)) // Util func
	if pathErr != nil {
		hoverLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, file.Version, line, col)
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

	// Format hover content
	hoverContent := formatObjectForHover(analysisInfo.IdentifierObject, analysisInfo, hoverLogger) // From helpers_hover.go
	if hoverContent == "" {
		hoverLogger.Debug("No hover content generated for identifier", "identifier", analysisInfo.IdentifierObject.Name())
		return nil, nil
	}

	// Determine hover range
	var hoverRange *LSPRange
	if analysisInfo.TargetFileSet != nil {
		lspRange, rangeErr := nodeRangeToLSPRange(analysisInfo.TargetFileSet, analysisInfo.IdentifierAtCursor, file.Content, hoverLogger) // From lsp_protocol.go
		if rangeErr == nil {
			hoverRange = lspRange
		} else {
			hoverLogger.Warn("Could not determine range for hover identifier", "error", rangeErr)
		}
	}

	// Determine markup kind
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

	line, col, _, posErr := LspPositionToBytePosition(file.Content, lspPos) // Util func
	if posErr != nil {
		defLogger.Error("Failed to convert LSP position to byte position", "error", posErr)
		return nil, nil // Return nil result
	}
	defLogger = defLogger.With("go_line", line, "go_col", col)

	absPath, pathErr := ValidateAndGetFilePath(string(uri)) // Util func
	if pathErr != nil {
		defLogger.Error("Invalid file URI", "error", pathErr)
		return nil, fmt.Errorf("invalid file URI: %w", pathErr)
	}

	analysisCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, file.Version, line, col)
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

	// Find the token.File corresponding to the definition position
	defFile := analysisInfo.TargetFileSet.File(defPos)
	if defFile == nil {
		defLogger.Error("Could not find token.File for definition position", "identifier", obj.Name(), "pos", defPos)
		return nil, nil
	}

	// Read definition file content for accurate LSP position conversion
	defFileContent, readErr := os.ReadFile(defFile.Name())
	if readErr != nil {
		defLogger.Error("Failed to read definition file content", "path", defFile.Name(), "error", readErr)
		s.sendShowMessage(MessageTypeWarning, fmt.Sprintf("Could not read definition file: %s", defFile.Name()))
		return nil, nil
	}

	// Convert the definition position to an LSP Location
	location, locErr := tokenPosToLSPLocation(defFile, defPos, defFileContent, defLogger) // From lsp_protocol.go
	if locErr != nil {
		defLogger.Error("Failed to convert definition position to LSP Location", "identifier", obj.Name(), "error", locErr)
		return nil, nil
	}

	defLogger.Info("Definition found", "identifier", obj.Name(), "location_uri", location.URI, "location_line", location.Range.Start.Line)
	return []Location{*location}, nil // Return as slice
}

func (s *Server) handleDidChangeConfiguration(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidChangeConfigurationParams) (any, error) {
	s.logger.Info("Handling workspace/didChangeConfiguration")

	var changedSettings struct {
		DeepComplete FileConfig `json:"deepcomplete"` // Use FileConfig from deepcomplete_types.go
	}

	if err := json.Unmarshal(params.Settings, &changedSettings); err != nil {
		s.logger.Error("Failed to unmarshal workspace/didChangeConfiguration settings", "error", err, "raw_settings", string(params.Settings))
		var directFileCfg FileConfig
		if directErr := json.Unmarshal(params.Settings, &directFileCfg); directErr == nil {
			s.logger.Info("Successfully unmarshalled settings directly into FileConfig")
			changedSettings.DeepComplete = directFileCfg
		} else {
			s.logger.Error("Also failed to unmarshal settings directly into FileConfig", "direct_error", directErr)
			return nil, nil
		}
	}

	newConfig := s.completer.GetCurrentConfig()
	fileCfg := changedSettings.DeepComplete
	mergedFields := 0

	// Merge non-nil fields from received settings
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
		if err := s.completer.UpdateConfig(newConfig); err != nil {
			s.logger.Error("Failed to apply updated configuration", "error", err)
			s.sendShowMessage(MessageTypeError, fmt.Sprintf("Failed to apply configuration update: %v", err))
		} else {
			s.config = s.completer.GetCurrentConfig() // Update server's local copy
			s.logger.Info("Server configuration updated successfully via workspace/didChangeConfiguration")
			// Attempt to update logger level
			newLevel, parseErr := ParseLogLevel(s.config.LogLevel) // Util func from deepcomplete_utils.go
			if parseErr == nil {
				s.logger.Info("Attempting to update logger level (implementation specific)", "new_level", newLevel)
				// Actual logger update logic depends on implementation
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
	ctx := context.Background()
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
		Version:     version,
		Diagnostics: diagnostics,
	}
	ctx := context.Background()
	if err := s.conn.Notify(ctx, "textDocument/publishDiagnostics", params); err != nil {
		s.logger.Error("Failed to send textDocument/publishDiagnostics notification", "error", err, "uri", uri, "diagnostic_count", len(diagnostics))
	} else {
		s.logger.Info("Published diagnostics", "uri", uri, "diagnostic_count", len(diagnostics), "version", version)
	}
}

// triggerDiagnostics performs analysis and publishes diagnostics. Requires absPath.
func (s *Server) triggerDiagnostics(uri DocumentURI, version int, content []byte, absPath string) {
	diagLogger := s.logger.With("uri", uri, "version", version, "absPath", absPath, "operation", "triggerDiagnostics")
	diagLogger.Info("Triggering background analysis for diagnostics")

	analysisCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Analyze using the absolute path, use placeholder line/col (1,1)
	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, version, 1, 1)

	lspDiagnostics := []LspDiagnostic{}
	if analysisInfo != nil && len(analysisInfo.Diagnostics) > 0 {
		diagLogger.Debug("Converting internal diagnostics to LSP format", "count", len(analysisInfo.Diagnostics))
		for _, diag := range analysisInfo.Diagnostics {
			// Convert internal range to LSP range
			lspRange, err := internalRangeToLSPRange(content, diag.Range, diagLogger) // Uses helper below
			if err != nil {
				diagLogger.Warn("Failed to convert diagnostic range, skipping diagnostic", "internal_range", diag.Range, "error", err, "message", diag.Message)
				continue
			}
			lspDiagnostics = append(lspDiagnostics, LspDiagnostic{
				Range:    *lspRange,
				Severity: mapInternalSeverityToLSP(diag.Severity), // Uses helper below
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
		diagLogger.Error("Fatal error during diagnostic analysis", "error", analysisErr)
	} else if analysisErr != nil {
		diagLogger.Warn("Analysis for diagnostics completed with non-fatal errors", "error", analysisErr)
	}

	s.publishDiagnostics(uri, &version, lspDiagnostics)
}

// internalRangeToLSPRange converts internal byte-offset range to LSP UTF-16 range.
func internalRangeToLSPRange(content []byte, internalRange Range, logger *slog.Logger) (*LSPRange, error) {
	if content == nil {
		return nil, errors.New("cannot convert range: content is nil")
	}
	startByteOffset := internalRange.Start.Character
	endByteOffset := internalRange.End.Character
	contentLen := len(content)

	if startByteOffset < 0 || endByteOffset < startByteOffset || endByteOffset > contentLen {
		return nil, fmt.Errorf("invalid internal byte offset range: start=%d, end=%d, content_len=%d", startByteOffset, endByteOffset, contentLen)
	}

	startLine, startChar, startErr := byteOffsetToLSPPosition(content, startByteOffset, logger) // Util func
	if startErr != nil {
		return nil, fmt.Errorf("failed converting start offset %d: %w", startByteOffset, startErr)
	}
	endLine, endChar, endErr := byteOffsetToLSPPosition(content, endByteOffset, logger) // Util func
	if endErr != nil {
		return nil, fmt.Errorf("failed converting end offset %d: %w", endByteOffset, endErr)
	}

	// Basic validation: ensure start is not after end
	if startLine > endLine || (startLine == endLine && startChar > endChar) {
		logger.Warn("Calculated invalid LSP range (end < start), adjusting end to start", "start_line", startLine, "start_char", startChar, "end_line", endLine, "end_char", endChar)
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

	// Publish cache metrics if available
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
	requests map[jsonrpc2.ID]context.CancelFunc
}

// NewRequestTracker creates a new tracker.
func NewRequestTracker() *RequestTracker {
	return &RequestTracker{
		requests: make(map[jsonrpc2.ID]context.CancelFunc),
	}
}

// Add registers a request ID and its associated context's cancel function.
func (rt *RequestTracker) Add(id jsonrpc2.ID, ctx context.Context) {
	if id == (jsonrpc2.ID{}) {
		return
	} // Ignore notifications
	rt.mu.Lock()
	defer rt.mu.Unlock()
	reqCtx, cancel := context.WithCancel(ctx)
	rt.requests[id] = cancel
	_ = reqCtx // Avoid unused variable error
}

// Remove deregisters a request ID.
func (rt *RequestTracker) Remove(id jsonrpc2.ID) {
	if id == (jsonrpc2.ID{}) {
		return
	} // Ignore notifications
	rt.mu.Lock()
	defer rt.mu.Unlock()
	delete(rt.requests, id)
}

// Cancel finds the cancel function for a request ID and calls it.
func (rt *RequestTracker) Cancel(id jsonrpc2.ID) {
	if id == (jsonrpc2.ID{}) { // Ignore notifications
		slog.Debug("Cancel request ignored for unset ID")
		return
	}
	rt.mu.Lock()
	cancel, found := rt.requests[id]
	if found {
		delete(rt.requests, id) // Remove immediately
	}
	rt.mu.Unlock()

	if found {
		slog.Debug("Calling cancel function for request", "id", id)
		cancel() // Call outside lock
	} else {
		slog.Debug("Cancel function not found for request ID", "id", id)
	}
}

// Count returns the number of currently tracked requests.
func (rt *RequestTracker) Count() int {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return len(rt.requests)
}
