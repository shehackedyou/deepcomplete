// deepcomplete/lsp_server.go
// Implements the main LSP server structure and request routing.
package deepcomplete

import (
	"context"
	"encoding/json"
	"errors"
	"expvar" // For metrics publishing
	"fmt"
	"io"
	"log/slog"
	"runtime"
	"runtime/debug" // For panic recovery
	"sync"
	"time"

	"github.com/sourcegraph/jsonrpc2"
)

// ============================================================================
// LSP Server Implementation
// ============================================================================

// Server represents the LSP server instance.
type Server struct {
	conn      *jsonrpc2.Conn
	logger    *slog.Logger
	completer *DeepCompleter // The core completion service
	files     map[DocumentURI]*OpenFile
	filesMu   sync.RWMutex
	// config Config // Removed: Get config directly from completer
	clientCaps     ClientCapabilities
	serverInfo     *ServerInfo
	initParams     *InitializeParams
	requestTracker *RequestTracker // Tracks active requests for cancellation
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
		completer: completer, // Completer holds the canonical config now
		files:     make(map[DocumentURI]*OpenFile),
		// config:    completer.GetCurrentConfig(), // Removed
		serverInfo: &ServerInfo{
			Name:    "DeepComplete LSP",
			Version: version,
		},
		requestTracker: NewRequestTracker(logger),
	}
	publishExpvarMetrics(s) // Publish metrics on startup
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

// handle routes incoming LSP requests/notifications to appropriate methods on the Server.
// Implementations of these methods are in lsp_handlers_*.go files.
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
			s.logger.Error("Panic recovered in handler", "panic_value", r, "stack", stack, "method", req.Method, "req_id", req.ID)

			panicMsg := fmt.Sprintf("Panic: %v", r)
			panicData, marshalErr := json.Marshal(panicMsg)
			if marshalErr != nil {
				s.logger.Error("Failed to marshal panic message for error data", "error", marshalErr)
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

		select {
		case <-ctx.Done():
			methodLogger.Warn("Request context cancelled before processing started", "error", ctx.Err())
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcRequestCancelled), Message: "Request cancelled"}
		default: // Continue processing
		}
	}

	// Helper to unmarshal params
	unmarshalParams := func(target any) error {
		if req.Params == nil {
			return errors.New("params field is null")
		}
		return json.Unmarshal(*req.Params, target)
	}

	// Route requests to the corresponding Server methods (defined in lsp_handlers_*.go)
	switch req.Method {
	case "initialize":
		var params InitializeParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal initialize params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid initialize params: %v", err)}
		}
		s.clientCaps = params.Capabilities
		s.initParams = &params
		return s.handleInitialize(ctx, conn, req, params, methodLogger)

	case "initialized":
		methodLogger.Info("Client initialized notification received")
		return nil, nil

	case "shutdown":
		methodLogger.Info("Shutdown request received")
		return s.handleShutdown(ctx, conn, req, methodLogger)

	case "exit":
		methodLogger.Info("Exit notification received")
		return s.handleExit(ctx, conn, req, methodLogger)

	case "textDocument/didOpen":
		var params DidOpenTextDocumentParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal didOpen params", "error", err)
			return nil, nil
		}
		return s.handleDidOpen(ctx, conn, req, params, methodLogger)

	case "textDocument/didChange":
		var params DidChangeTextDocumentParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal didChange params", "error", err)
			return nil, nil
		}
		return s.handleDidChange(ctx, conn, req, params, methodLogger)

	case "textDocument/didClose":
		var params DidCloseTextDocumentParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal didClose params", "error", err)
			return nil, nil
		}
		return s.handleDidClose(ctx, conn, req, params, methodLogger)

	case "textDocument/completion":
		var params CompletionParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal completion params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid completion params: %v", err)}
		}
		return s.handleCompletion(ctx, conn, req, params, methodLogger)

	case "textDocument/hover":
		var params HoverParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal hover params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid hover params: %v", err)}
		}
		return s.handleHover(ctx, conn, req, params, methodLogger)

	case "textDocument/definition":
		var params DefinitionParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal definition params", "error", err)
			return nil, &jsonrpc2.Error{Code: int64(JsonRpcInvalidParams), Message: fmt.Sprintf("Invalid definition params: %v", err)}
		}
		return s.handleDefinition(ctx, conn, req, params, methodLogger)

	case "workspace/didChangeConfiguration":
		var params DidChangeConfigurationParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal didChangeConfiguration params", "error", err)
			return nil, nil
		}
		return s.handleDidChangeConfiguration(ctx, conn, req, params, methodLogger)

	case "$/cancelRequest":
		var params CancelParams
		if err := unmarshalParams(&params); err != nil {
			methodLogger.Error("Failed to unmarshal cancelRequest params", "error", err)
			return nil, nil
		}
		var cancelID jsonrpc2.ID
		switch idVal := params.ID.(type) {
		case float64:
			cancelID = jsonrpc2.ID{Num: uint64(idVal)}
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
// LSP Notification Sending Helpers
// ============================================================================

// sendShowMessage sends a window/showMessage notification. Uses the server's logger.
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

// publishDiagnostics sends a textDocument/publishDiagnostics notification.
func (s *Server) publishDiagnostics(uri DocumentURI, version *int, diagnostics []LspDiagnostic, logger *slog.Logger) {
	if logger == nil {
		logger = s.logger
	}
	if s.conn == nil {
		logger.Warn("Cannot publish diagnostics: connection is nil", "uri", uri)
		return
	}
	params := PublishDiagnosticsParams{
		URI:         uri,
		Version:     version,
		Diagnostics: diagnostics,
	}
	ctx := context.Background()
	if err := s.conn.Notify(ctx, "textDocument/publishDiagnostics", params); err != nil {
		logger.Error("Failed to send textDocument/publishDiagnostics notification", "error", err, "uri", uri, "diagnostic_count", len(diagnostics))
	} else {
		logger.Info("Published diagnostics", "uri", uri, "diagnostic_count", len(diagnostics), "version", version)
	}
}

// triggerDiagnostics performs analysis and publishes diagnostics. Requires absPath.
// Accepts a logger instance.
func (s *Server) triggerDiagnostics(uri DocumentURI, version int, content []byte, absPath string, logger *slog.Logger) {
	if logger == nil {
		logger = s.logger
	}
	diagLogger := logger.With("uri", uri, "version", version, "absPath", absPath, "operation", "triggerDiagnostics")
	diagLogger.Info("Triggering background analysis for diagnostics")
	analysisStart := time.Now()

	analysisCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	analysisInfo, analysisErr := s.completer.analyzer.Analyze(analysisCtx, absPath, version, 1, 1)
	diagLogger.Info("Analysis for diagnostics completed", "duration", time.Since(analysisStart))

	// --- Check if file version changed during analysis ---
	s.filesMu.RLock()
	currentFile, exists := s.files[uri]
	s.filesMu.RUnlock()
	// Only publish if the file still exists in our map AND the version matches the one we analyzed
	if !exists || currentFile.Version != version {
		diagLogger.Warn("Skipping diagnostic publishing; file version changed or file closed during analysis", "analysis_version", version, "current_version", currentFile.Version, "exists", exists)
		return
	}
	// --- End Version Check ---

	lspDiagnostics := []LspDiagnostic{}
	if analysisInfo != nil && len(analysisInfo.Diagnostics) > 0 {
		diagLogger.Debug("Converting internal diagnostics to LSP format", "count", len(analysisInfo.Diagnostics))
		for _, diag := range analysisInfo.Diagnostics {
			lspRange, err := internalRangeToLSPRange(content, diag.Range, diagLogger)
			if err != nil {
				diagLogger.Warn("Failed to convert diagnostic range, skipping diagnostic", "internal_range", diag.Range, "error", err, "message", diag.Message)
				continue
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

	if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
		diagLogger.Error("Fatal error during diagnostic analysis", "error", analysisErr)
	} else if analysisErr != nil {
		diagLogger.Warn("Analysis for diagnostics completed with non-fatal errors", "error", analysisErr)
	}

	// Publish diagnostics for the correct version
	s.publishDiagnostics(uri, &version, lspDiagnostics, diagLogger)
}

// internalRangeToLSPRange converts internal byte-offset range to LSP UTF-16 range.
func internalRangeToLSPRange(content []byte, internalRange Range, logger *slog.Logger) (*LSPRange, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if content == nil {
		return nil, errors.New("cannot convert range: content is nil")
	}
	startByteOffset := internalRange.Start.Character
	endByteOffset := internalRange.End.Character
	contentLen := len(content)

	if startByteOffset < 0 || endByteOffset < startByteOffset || endByteOffset > contentLen {
		return nil, fmt.Errorf("invalid internal byte offset range: start=%d, end=%d, content_len=%d", startByteOffset, endByteOffset, contentLen)
	}

	startLine, startChar, startErr := byteOffsetToLSPPosition(content, startByteOffset, logger)
	if startErr != nil {
		return nil, fmt.Errorf("failed converting start offset %d: %w", startByteOffset, startErr)
	}
	endLine, endChar, endErr := byteOffsetToLSPPosition(content, endByteOffset, logger)
	if endErr != nil {
		return nil, fmt.Errorf("failed converting end offset %d: %w", endByteOffset, endErr)
	}

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
// Metrics Publishing
// ============================================================================

// publishExpvarMetrics publishes server metrics using the expvar package.
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
	expvar.Publish("memory.totalAllocBytes", expvar.Func(func() any {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return m.TotalAlloc
	}))
	expvar.Publish("memory.heapAllocBytes", expvar.Func(func() any {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return m.HeapAlloc
	}))
	expvar.Publish("memory.numGC", expvar.Func(func() any {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return m.NumGC
	}))

	expvar.Publish("lsp.openFiles", expvar.Func(func() any {
		s.filesMu.RLock()
		defer s.filesMu.RUnlock()
		return len(s.files)
	}))
	expvar.Publish("lsp.pendingRequests", expvar.Func(func() any { return s.requestTracker.Count() }))

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
		expvar.Publish("cache.memory.ratio", expvar.Func(func() any {
			m := s.completer.analyzer.GetMemoryCacheMetrics()
			if m != nil {
				return fmt.Sprintf("%.4f", m.Ratio())
			}
			return "0.0000"
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
		expvar.Publish("cache.memory.keysUpdated", expvar.Func(func() any {
			m := s.completer.analyzer.GetMemoryCacheMetrics()
			if m != nil {
				return m.KeysUpdated()
			}
			return 0
		}))
	} else {
		expvar.Publish("cache.memory.hits", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.misses", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.ratio", expvar.Func(func() any { return "0.0000" }))
		expvar.Publish("cache.memory.costAdded", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.costEvicted", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.keysAdded", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.keysEvicted", expvar.Func(func() any { return 0 }))
		expvar.Publish("cache.memory.keysUpdated", expvar.Func(func() any { return 0 }))
	}
	s.logger.Info("Expvar metrics published")
}

// ============================================================================
// Request Cancellation Tracker
// ============================================================================

// RequestTracker manages cancellation contexts for ongoing LSP requests.
type RequestTracker struct {
	mu       sync.Mutex
	requests map[jsonrpc2.ID]context.CancelFunc
	logger   *slog.Logger // Use logger passed during creation
}

// NewRequestTracker creates a new tracker.
func NewRequestTracker(logger *slog.Logger) *RequestTracker {
	if logger == nil {
		logger = slog.Default() // Fallback
	}
	return &RequestTracker{
		requests: make(map[jsonrpc2.ID]context.CancelFunc),
		logger:   logger.With("component", "RequestTracker"), // Add component context
	}
}

// Add registers a request ID and its associated context's cancel function.
func (rt *RequestTracker) Add(id jsonrpc2.ID, ctx context.Context) {
	if id == (jsonrpc2.ID{}) {
		rt.logger.Debug("Ignoring attempt to track notification (no ID)")
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if _, exists := rt.requests[id]; exists {
		rt.logger.Warn("Attempted to add already tracked request ID", "id", id)
		return
	}
	reqCtx, cancel := context.WithCancel(context.Background())
	rt.requests[id] = cancel
	rt.logger.Debug("Added request to tracker", "id", id)

	go func() {
		select {
		case <-ctx.Done():
			rt.logger.Warn("Original request context done, ensuring tracked request is cancelled", "id", id, "reason", ctx.Err())
			rt.Cancel(id)
			rt.Remove(id)
		case <-reqCtx.Done():
			rt.logger.Debug("Tracked request context cancelled", "id", id)
		}
	}()
}

// Remove deregisters a request ID. Should be called when a request handler finishes.
func (rt *RequestTracker) Remove(id jsonrpc2.ID) {
	if id == (jsonrpc2.ID{}) {
		return // Ignore notifications
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if cancel, ok := rt.requests[id]; ok {
		rt.logger.Debug("Removing request from tracker", "id", id)
		cancel() // Ensure cancellation function is called
		delete(rt.requests, id)
	} else {
		rt.logger.Debug("Attempted to remove request ID not found in tracker", "id", id)
	}
}

// Cancel finds the cancel function for a request ID and calls it.
func (rt *RequestTracker) Cancel(id jsonrpc2.ID) {
	if id == (jsonrpc2.ID{}) {
		rt.logger.Debug("Cancel request ignored for unset ID")
		return
	}
	rt.mu.Lock()
	cancel, found := rt.requests[id]
	rt.mu.Unlock()

	if found {
		rt.logger.Info("Calling cancel function for request via $/cancelRequest", "id", id)
		cancel()
	} else {
		rt.logger.Debug("Cancel function not found for request ID (already removed or never added?)", "id", id)
	}
}

// Count returns the number of currently tracked requests.
func (rt *RequestTracker) Count() int {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return len(rt.requests)
}
