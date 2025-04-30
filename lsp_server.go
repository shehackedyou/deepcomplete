// deepcomplete/lsp_server.go
// Implements the LSP server logic, moved from cmd/deepcomplete-lsp/main.go
package deepcomplete

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"expvar"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/match"
)

// ============================================================================
// Server Definition & Internal Types
// ============================================================================

// VersionedContent stores document content along with its version and access time.
type VersionedContent struct {
	Content    []byte
	Version    int
	LastAccess time.Time
}

// --- Event Definitions ---
// Event represents a parsed LSP message (request or notification).
type Event interface {
	GetType() string               // Returns the LSP method name.
	GetRequestID() any             // Returns the request ID (nil for notifications).
	GetRawParams() json.RawMessage // Returns the raw parameters
}

// BaseEvent provides common fields for events.
type BaseEvent struct {
	Type      string
	RequestID any // nil for notifications
	RawParams json.RawMessage
}

func (e BaseEvent) GetType() string               { return e.Type }
func (e BaseEvent) GetRequestID() any             { return e.RequestID }
func (e BaseEvent) GetRawParams() json.RawMessage { return e.RawParams }

// --- Request Events ---
type InitializeRequestEvent struct{ BaseEvent }
type ShutdownRequestEvent struct{ BaseEvent }
type CompletionRequestEvent struct{ BaseEvent }
type HoverRequestEvent struct{ BaseEvent }
type DefinitionRequestEvent struct{ BaseEvent }

// --- Notification Events ---
type InitializedNotificationEvent struct{ BaseEvent }
type ExitNotificationEvent struct{ BaseEvent }
type DidOpenNotificationEvent struct{ BaseEvent }
type DidCloseNotificationEvent struct{ BaseEvent }
type DidChangeNotificationEvent struct{ BaseEvent }
type DidChangeConfigurationNotificationEvent struct{ BaseEvent }
type CancelRequestEvent struct{ BaseEvent }

// --- Internal/Error Event ---
type UnknownEvent struct {
	BaseEvent
	ParseError error
}

// --- Middleware & Handler Types ---
type HandlerFunc func(ctx context.Context, event Event, logger *slog.Logger) (result any, err *ErrorObject)
type MiddlewareFunc func(next HandlerFunc) HandlerFunc

// --- Work Item Definition ---
type ResponseWorkItem struct {
	RequestID any
	Result    any
	Error     *ErrorObject
}

// --- Global expvar variables (Defined ONCE at package level) ---
// These are used across the server instance to track metrics.
var (
	eventsReceived           = expvar.NewInt("lsp.eventsReceived")
	eventsIgnored            = expvar.NewInt("lsp.eventsIgnored")
	requestsDispatched       = expvar.NewInt("lsp.requestsDispatched")
	notificationsHandled     = expvar.NewInt("lsp.notificationsHandled")
	responsesSent            = expvar.NewInt("lsp.responsesSent")
	notificationsSent        = expvar.NewInt("lsp.notificationsSent")
	errorsReported           = expvar.NewInt("lsp.errorsReported")
	panicsRecovered          = expvar.NewInt("lsp.panicsRecovered")
	workerSemaphoreAcqTry    = expvar.NewInt("lsp.workerSemaphoreAcqTry")
	workerSemaphoreAcqOk     = expvar.NewInt("lsp.workerSemaphoreAcqOk")
	workerSemaphoreTimeout   = expvar.NewInt("lsp.workerSemaphoreTimeout")
	responseQueueTimeout     = expvar.NewInt("lsp.responseQueueTimeout")
	notificationQueueTimeout = expvar.NewInt("lsp.notificationQueueTimeout")
	docsEvicted              = expvar.NewInt("lsp.docsEvicted")
	diagnosticsPublished     = expvar.NewInt("lsp.diagnosticsPublished")
)

// Server encapsulates the state and methods of the LSP server.
type Server struct {
	completer *DeepCompleter // Core completion/analysis logic
	logger    *slog.Logger   // Server's logger instance

	// Server State
	documentStore          map[DocumentURI]VersionedContent // Uses type from lsp_protocol.go
	docStoreMutex          sync.RWMutex
	clientSupportsSnippets bool
	appVersion             string // Set during initialization

	// Concurrency & Communication
	eventQueue        chan Event
	responseQueue     chan ResponseWorkItem
	notificationQueue chan NotificationMessage
	shutdownChan      chan struct{}
	wg                sync.WaitGroup
	activeRequests    map[any]context.CancelFunc
	activeRequestMut  sync.Mutex
	workerSemaphore   chan struct{}

	// Configuration
	maxOpenDocuments     int
	maxConcurrentWorkers int
	responseSendTimeout  time.Duration
	notifySendTimeout    time.Duration
	analysisTimeout      time.Duration

	// Monitoring - uses global expvars defined above
}

// NewServer creates a new LSP Server instance.
func NewServer(completer *DeepCompleter, logger *slog.Logger, appVersion string) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	workers := runtime.NumCPU()
	if workers < 2 {
		workers = 2
	}

	s := &Server{
		completer:            completer,
		logger:               logger,
		appVersion:           appVersion,
		documentStore:        make(map[DocumentURI]VersionedContent),
		activeRequests:       make(map[any]context.CancelFunc),
		eventQueue:           make(chan Event, 100),
		responseQueue:        make(chan ResponseWorkItem, 100),
		notificationQueue:    make(chan NotificationMessage, 20),
		shutdownChan:         make(chan struct{}),
		workerSemaphore:      make(chan struct{}, workers),
		maxOpenDocuments:     100,
		maxConcurrentWorkers: workers,
		responseSendTimeout:  2 * time.Second,
		notifySendTimeout:    1 * time.Second,
		analysisTimeout:      30 * time.Second,
	}
	s.publishExpvarMetrics() // Publish metrics after server struct is created
	return s
}

// publishExpvarMetrics sets up expvar metrics publishing for this server instance.
func (s *Server) publishExpvarMetrics() {
	// Use a map to prevent accidental reuse panics during development/refactoring
	publishedMetrics := make(map[string]bool)
	publish := func(name string, v expvar.Var) {
		// Check if already published. If expvar panics here, it means
		// NewServer is somehow called multiple times, which shouldn't happen.
		if _, exists := publishedMetrics[name]; exists {
			s.logger.Warn("Attempting to publish expvar metric again, skipping.", "name", name)
			return
		}
		// Check if the name is already globally registered by another instance/package
		// Note: expvar.Get is not ideal as it returns nil if not found, but we can try.
		// A more robust approach might involve a global registry if multiple Server instances
		// were expected, but for a single LSP server, this check helps debugging.
		if expvar.Get(name) != nil {
			s.logger.Error("Expvar metric name already registered globally before Server publish.", "name", name)
			// Potentially panic here or just log the error depending on desired strictness
			// panic(fmt.Sprintf("Reuse of exported var name: %s", name))
			return // Avoid panic, just log and skip republishing
		}

		expvar.Publish(name, v)
		publishedMetrics[name] = true
	}

	// Publish metrics that depend on the server instance state
	publish("lsp.eventQueueLength", expvar.Func(func() any {
		if s.eventQueue == nil {
			return 0
		}
		return len(s.eventQueue)
	}))
	publish("lsp.responseQueueLength", expvar.Func(func() any {
		if s.responseQueue == nil {
			return 0
		}
		return len(s.responseQueue)
	}))
	publish("lsp.notificationQueueLength", expvar.Func(func() any {
		if s.notificationQueue == nil {
			return 0
		}
		return len(s.notificationQueue)
	}))
	publish("lsp.activeRequests", expvar.Func(func() any {
		s.activeRequestMut.Lock()
		defer s.activeRequestMut.Unlock()
		// Check if map is nil defensively
		if s.activeRequests == nil {
			return 0
		}
		return len(s.activeRequests)
	}))
	publish("lsp.activeWorkers", expvar.Func(func() any {
		if s.workerSemaphore == nil {
			return 0
		}
		return s.maxConcurrentWorkers - len(s.workerSemaphore)
	}))
	publish("lsp.openDocuments", expvar.Func(func() any {
		s.docStoreMutex.RLock()
		defer s.docStoreMutex.RUnlock()
		// Check if map is nil defensively
		if s.documentStore == nil {
			return 0
		}
		return len(s.documentStore)
	}))

	// Publish the global counter variables (defined at package level)
	// These are safe to publish directly as they are expvar.Int types.
	// No need to call publish() if they are already globally defined via NewInt.
	// The expvar package handles making them available at the /debug/vars endpoint.
	// We just need to ensure they are defined *once* globally.

	// Publish Ristretto metrics if cache is enabled
	if s.completer != nil {
		if analyzer := s.completer.GetAnalyzer(); analyzer != nil && analyzer.MemoryCacheEnabled() {
			metrics := analyzer.GetMemoryCacheMetrics()
			if metrics != nil {
				// Publish functions that retrieve the *current* metric value
				publish("lsp.ristrettoHits", expvar.Func(func() any { return metrics.Hits() }))
				publish("lsp.ristrettoMisses", expvar.Func(func() any { return metrics.Misses() }))
				publish("lsp.ristrettoKeysAdded", expvar.Func(func() any { return metrics.KeysAdded() }))
				publish("lsp.ristrettoCostAdded", expvar.Func(func() any { return metrics.CostAdded() }))
				publish("lsp.ristrettoCostEvicted", expvar.Func(func() any { return metrics.CostEvicted() }))
				publish("lsp.ristrettoRatio", expvar.Func(func() any { return metrics.Ratio() }))
			}
		}
	}
	s.logger.Debug("Expvar metrics publishing setup complete.")
}

// Run starts the LSP server's main loops (reader, dispatcher, writer).
// It blocks until the server is shut down.
func (s *Server) Run(reader io.Reader, writer io.Writer) {
	s.logger.Info("Starting LSP Server loops...")

	s.wg.Add(3)
	go s.runReader(reader)
	go s.runDispatcher()
	go s.runResponseWriter(writer)
	s.logger.Info("Started Reader, Dispatcher, and Writer goroutines.")

	<-s.shutdownChan
	s.logger.Info("Shutdown signal received in Server.Run.")

	s.logger.Info("Waiting for server goroutines to finish...")
	s.wg.Wait()
	s.logger.Info("All server goroutines finished.")
}

// ============================================================================
// Event Loop Goroutines (Reader, Dispatcher, Writer)
// ============================================================================

// runReader reads messages from the client, parses them, and sends them to the eventQueue.
func (s *Server) runReader(r io.Reader) {
	defer s.wg.Done()
	defer func() {
		if r := recover(); r != nil {
			panicsRecovered.Add(1) // Use global expvar
			s.logger.Error("PANIC in runReader", "error", r, "stack", string(debug.Stack()))
		}
		s.logger.Info("Reader goroutine stopped.")
	}()

	reader := bufio.NewReader(r)
	for {
		select {
		case <-s.shutdownChan:
			return
		default:
			content, err := s.readMessage(reader) // Use server method
			if err != nil {
				if errors.Is(err, io.EOF) {
					s.logger.Info("Client closed connection (EOF). Stopping reader and signaling shutdown.")
					s.signalShutdown() // Use server method
					return
				}
				s.logger.Error("Error reading message", "error", err)
				time.Sleep(100 * time.Millisecond)
				continue
			}

			eventsReceived.Add(1) // Use global expvar
			s.logger.Debug("Received raw message", "bytes", len(content))
			event := s.parseMessageToEvent(content) // Use server method
			if event == nil {
				eventsIgnored.Add(1) // Use global expvar
				continue
			}
			logAttrs := []any{slog.String("type", event.GetType())}
			if reqID := event.GetRequestID(); reqID != nil {
				logAttrs = append(logAttrs, slog.Any("requestID", reqID))
			}
			s.logger.Debug("Parsed event", logAttrs...)

			select {
			case s.eventQueue <- event:
				// Event sent
			case <-s.shutdownChan:
				s.logger.Warn("Shutdown signaled while sending event to queue.")
				return
			case <-time.After(5 * time.Second):
				s.logger.Error("Timeout sending event to eventQueue (dispatcher slow/stuck?)", "eventType", event.GetType())
			}
		}
	}
}

// runDispatcher reads events, applies middleware, and spawns workers.
func (s *Server) runDispatcher() {
	defer s.wg.Done()
	defer func() {
		if r := recover(); r != nil {
			panicsRecovered.Add(1) // Use global expvar
			s.logger.Error("PANIC in runDispatcher", "error", r, "stack", string(debug.Stack()))
		}
		s.logger.Info("Dispatcher goroutine stopped.")
	}()

	serverCtx, cancelServer := context.WithCancel(context.Background())
	defer cancelServer()

	middlewares := []MiddlewareFunc{
		s.panicRecoveryMiddleware, // Use server methods for middleware
		s.loggingMiddleware,
		s.cancellationMiddleware,
	}

	for {
		select {
		case <-s.shutdownChan:
			s.logger.Info("Dispatcher received shutdown signal.")
			cancelServer()
			return
		case event, ok := <-s.eventQueue:
			if !ok {
				s.logger.Info("Event queue closed. Stopping dispatcher.")
				cancelServer()
				return
			}

			eventLogger := s.logger.With(
				"requestID", event.GetRequestID(),
				"eventType", fmt.Sprintf("%T", event),
			)
			eventLogger.Debug("Dispatcher received event")

			var baseHandler HandlerFunc
			isResourceIntensive := false
			isNotification := event.GetRequestID() == nil
			handlerName := ""

			switch ev := event.(type) {
			case InitializeRequestEvent:
				baseHandler = s.handleInitialize
				handlerName = "Initialize"
			case ShutdownRequestEvent:
				baseHandler = s.handleShutdown
				handlerName = "Shutdown"
			case CompletionRequestEvent:
				baseHandler = s.handleCompletion
				handlerName = "Completion"
				isResourceIntensive = true
			case HoverRequestEvent:
				baseHandler = s.handleHover
				handlerName = "Hover"
				isResourceIntensive = true
			case DefinitionRequestEvent:
				baseHandler = s.handleDefinition
				handlerName = "Definition"
				isResourceIntensive = true
			case InitializedNotificationEvent:
				handlerName = "Initialized"
				go s.handleInitializedNotification(serverCtx, ev, eventLogger.With("handler", handlerName))
				notificationsHandled.Add(1) // Use global expvar
				continue
			case ExitNotificationEvent:
				handlerName = "Exit"
				go s.handleExitNotification(serverCtx, ev, eventLogger.With("handler", handlerName))
				notificationsHandled.Add(1) // Use global expvar
				continue
			case DidOpenNotificationEvent:
				handlerName = "DidOpen"
				go s.handleDidOpenNotification(serverCtx, ev, eventLogger.With("handler", handlerName))
				notificationsHandled.Add(1) // Use global expvar
				continue
			case DidCloseNotificationEvent:
				handlerName = "DidClose"
				go s.handleDidCloseNotification(serverCtx, ev, eventLogger.With("handler", handlerName))
				notificationsHandled.Add(1) // Use global expvar
				continue
			case DidChangeNotificationEvent:
				handlerName = "DidChange"
				go s.handleDidChangeNotification(serverCtx, ev, eventLogger.With("handler", handlerName))
				notificationsHandled.Add(1) // Use global expvar
				continue
			case DidChangeConfigurationNotificationEvent:
				handlerName = "DidChangeConfiguration"
				go s.handleDidChangeConfigurationNotification(serverCtx, ev, eventLogger.With("handler", handlerName))
				notificationsHandled.Add(1) // Use global expvar
				continue
			case CancelRequestEvent:
				handlerName = "CancelRequest"
				go s.handleCancelRequestNotification(serverCtx, ev, eventLogger.With("handler", handlerName))
				notificationsHandled.Add(1) // Use global expvar
				continue
			case UnknownEvent:
				eventLogger.Warn("Dispatcher received UnknownEvent", "error", ev.ParseError)
				if reqID := event.GetRequestID(); reqID != nil {
					errorsReported.Add(1) // Use global expvar
					s.sendResponse(reqID, nil, &ErrorObject{Code: JsonRpcMethodNotFound, Message: fmt.Sprintf("Method not found: %s", event.GetType())}, serverCtx, eventLogger)
				}
				continue
			default:
				eventLogger.Warn("Dispatcher received unhandled event type")
				continue
			}

			if baseHandler != nil {
				requestsDispatched.Add(1) // Use global expvar
				wrappedHandler := baseHandler
				for i := len(middlewares) - 1; i >= 0; i-- {
					isCancellationMw := fmt.Sprintf("%p", middlewares[i]) == fmt.Sprintf("%p", s.cancellationMiddleware)
					if isNotification && isCancellationMw {
						continue
					}
					wrappedHandler = middlewares[i](wrappedHandler)
				}

				workerLogger := eventLogger.With("handler", handlerName)

				go func(handler HandlerFunc, reqEvent Event, reqLogger *slog.Logger, limit bool) {
					if limit {
						workerSemaphoreAcqTry.Add(1) // Use global expvar
						reqLogger.Debug("Acquiring worker semaphore slot...")
						select {
						case s.workerSemaphore <- struct{}{}:
							workerSemaphoreAcqOk.Add(1) // Use global expvar
							reqLogger.Debug("Worker semaphore slot acquired.")
							defer func() {
								<-s.workerSemaphore
								reqLogger.Debug("Worker semaphore slot released.")
							}()
						case <-serverCtx.Done():
							reqLogger.Warn("Server shutdown while waiting for worker semaphore.")
							if reqID := reqEvent.GetRequestID(); reqID != nil {
								s.sendResponse(reqID, nil, &ErrorObject{Code: JsonRpcRequestCancelled, Message: "Server shutting down"}, serverCtx, reqLogger)
							}
							return
						}
					} else {
						reqLogger.Debug("Skipping semaphore for non-intensive handler.")
					}

					res, errObj := handler(serverCtx, reqEvent, reqLogger)

					if reqID := reqEvent.GetRequestID(); reqID != nil {
						s.sendResponse(reqID, res, errObj, serverCtx, reqLogger)
					} else if errObj != nil {
						reqLogger.Error("Error occurred processing notification within middleware", "error", errObj)
						errorsReported.Add(1) // Use global expvar
					}
				}(wrappedHandler, event, workerLogger, isResourceIntensive)
			}
		}
	}
}

// runResponseWriter reads completed work items and notifications, writing them to the client.
func (s *Server) runResponseWriter(w io.Writer) {
	defer s.wg.Done()
	defer func() {
		if r := recover(); r != nil {
			panicsRecovered.Add(1) // Use global expvar
			s.logger.Error("PANIC in runResponseWriter", "error", r, "stack", string(debug.Stack()))
		}
		s.logger.Info("Response Writer goroutine stopped.")
	}()

	var writerMu sync.Mutex

	for {
		select {
		case <-s.shutdownChan:
			s.logger.Info("Response Writer received shutdown signal.")
			return
		case workItem, ok := <-s.responseQueue:
			if !ok {
				s.logger.Info("Response queue closed.")
				s.responseQueue = nil // Mark as closed
				if s.notificationQueue == nil {
					return
				}
				continue
			}
			response := ResponseMessage{
				JSONRPC: "2.0",
				ID:      workItem.RequestID,
				Result:  workItem.Result,
				Error:   workItem.Error,
			}
			writerMu.Lock()
			err := s.writeMessage(w, response) // Use server method
			writerMu.Unlock()
			if err != nil {
				s.logger.Error("Error writing response", "requestID", workItem.RequestID, "error", err)
			} else {
				responsesSent.Add(1) // Use global expvar
				s.logger.Debug("Successfully wrote response", "requestID", workItem.RequestID)
			}

		case notificationMsg, ok := <-s.notificationQueue:
			if !ok {
				s.logger.Info("Notification queue closed.")
				s.notificationQueue = nil // Mark as closed
				if s.responseQueue == nil {
					return
				}
				continue
			}
			writerMu.Lock()
			err := s.writeMessage(w, notificationMsg) // Use server method
			writerMu.Unlock()
			if err != nil {
				s.logger.Error("Error writing notification", "method", notificationMsg.Method, "error", err)
			} else {
				notificationsSent.Add(1) // Use global expvar
				s.logger.Debug("Successfully wrote notification", "method", notificationMsg.Method)
			}
		}
	}
}

// sendResponse is a helper to send a ResponseWorkItem to the queue with timeout.
func (s *Server) sendResponse(reqID any, result any, errObj *ErrorObject, serverCtx context.Context, logger *slog.Logger) {
	if reqID == nil {
		logger.Warn("Attempted to send response for notification")
		return
	}
	workItem := ResponseWorkItem{RequestID: reqID, Result: result, Error: errObj}
	select {
	case s.responseQueue <- workItem:
		if errObj != nil {
			errorsReported.Add(1) // Use global expvar
		}
		logger.Debug("Successfully sent response item to queue")
	case <-time.After(s.responseSendTimeout):
		responseQueueTimeout.Add(1) // Use global expvar
		logger.Error("Timeout sending response to queue (writer blocked?) - dropping response.", "requestID", workItem.RequestID, "timeout", s.responseSendTimeout)
		errorsReported.Add(1) // Use global expvar
	case <-serverCtx.Done():
		logger.Warn("Shutdown occurred while attempting to send response.", "requestID", workItem.RequestID)
	}
}

// signalShutdown safely closes the shutdown channel.
func (s *Server) signalShutdown() {
	select {
	case <-s.shutdownChan: // Already closed
	default:
		close(s.shutdownChan)
	}
}

// ============================================================================
// I/O & Parsing Helpers (Now methods on Server)
// ============================================================================

// readMessage reads a single JSON-RPC message based on Content-Length header.
func (s *Server) readMessage(reader *bufio.Reader) ([]byte, error) {
	var contentLength int = -1
	for {
		lineBytes, err := reader.ReadBytes('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil, io.EOF
			}
			return nil, fmt.Errorf("failed reading header line bytes: %w", err)
		}
		line := string(bytes.TrimSpace(lineBytes))
		if line == "" {
			break
		}

		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			valueStr := strings.TrimSpace(line[len("content-length:"):])
			_, err := fmt.Sscan(valueStr, &contentLength)
			if err != nil {
				s.logger.Warn("Failed to parse Content-Length header value", "header_line", line, "value_part", valueStr, "error", err)
				contentLength = -1
			}
		}
	}

	if contentLength < 0 {
		return nil, fmt.Errorf("missing or invalid Content-Length header found")
	}
	if contentLength == 0 {
		s.logger.Debug("Received message with Content-Length: 0")
		return []byte{}, nil
	}

	body := make([]byte, contentLength)
	n, err := io.ReadFull(reader, body)
	if err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			s.logger.Warn("Client disconnected before sending full message body", "expected", contentLength, "read", n)
			return nil, io.EOF
		}
		return nil, fmt.Errorf("failed reading message body (read %d/%d bytes): %w", n, contentLength, err)
	}
	return body, nil
}

// writeMessage encodes a message (Response or Notification) and writes it.
func (s *Server) writeMessage(writer io.Writer, message interface{}) error {
	if message == nil {
		s.logger.Error("Attempted to write nil message")
		return errors.New("cannot write nil message")
	}

	content, err := json.Marshal(message)
	if err != nil {
		s.logger.Error("Error marshalling message for output", "error", err, "messageType", fmt.Sprintf("%T", message))
		if respMsg, ok := message.(ResponseMessage); ok {
			errMsg := ResponseMessage{
				JSONRPC: "2.0",
				ID:      respMsg.ID,
				Error:   &ErrorObject{Code: JsonRpcInternalError, Message: "Failed to marshal server response"},
			}
			content, err = json.Marshal(errMsg)
			if err != nil {
				s.logger.Error("Failed to marshal generic error response", "error", err)
				return fmt.Errorf("failed to marshal even generic error response: %w", err)
			}
		} else {
			return fmt.Errorf("failed to marshal notification message: %w", err)
		}
	}

	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(content))
	s.logger.Debug("Sending message", "header", header, "content_length", len(content), "content_preview", firstN(string(content), 100))

	if _, err := writer.Write([]byte(header)); err != nil {
		s.logger.Error("Error writing message header", "error", err)
		return fmt.Errorf("error writing header: %w", err)
	}
	if _, err := writer.Write(content); err != nil {
		s.logger.Error("Error writing message content", "error", err)
		return fmt.Errorf("error writing content: %w", err)
	}
	return nil
}

// parseMessageToEvent parses raw message content into a specific Event struct.
func (s *Server) parseMessageToEvent(content []byte) Event {
	methodResult := gjson.GetBytes(content, "method")
	if !methodResult.Exists() {
		s.logger.Error("Received message without method", "content_preview", firstN(string(content), 100))
		return UnknownEvent{BaseEvent: BaseEvent{RawParams: content}, ParseError: errors.New("message missing method field")}
	}
	method := methodResult.String()
	idResult := gjson.GetBytes(content, "id")
	rawParams := json.RawMessage(gjson.GetBytes(content, "params").Raw)

	base := BaseEvent{Type: method, RawParams: rawParams}

	if idResult.Exists() { // Request
		base.RequestID = idResult.Value()
		switch method {
		case "initialize":
			return InitializeRequestEvent{BaseEvent: base}
		case "shutdown":
			return ShutdownRequestEvent{BaseEvent: base}
		case "textDocument/completion":
			return CompletionRequestEvent{BaseEvent: base}
		case "textDocument/hover":
			return HoverRequestEvent{BaseEvent: base}
		case "textDocument/definition":
			return DefinitionRequestEvent{BaseEvent: base}
		default:
			s.logger.Warn("Received unknown request method", "method", method, "requestID", base.RequestID)
			return UnknownEvent{BaseEvent: base, ParseError: fmt.Errorf("unknown request method: %s", method)}
		}
	} else { // Notification
		base.RequestID = nil
		switch method {
		case "initialized":
			return InitializedNotificationEvent{BaseEvent: base}
		case "exit":
			return ExitNotificationEvent{BaseEvent: base}
		case "textDocument/didOpen":
			return DidOpenNotificationEvent{BaseEvent: base}
		case "textDocument/didClose":
			return DidCloseNotificationEvent{BaseEvent: base}
		case "textDocument/didChange":
			return DidChangeNotificationEvent{BaseEvent: base}
		case "workspace/didChangeConfiguration":
			return DidChangeConfigurationNotificationEvent{BaseEvent: base}
		case "$/cancelRequest":
			return CancelRequestEvent{BaseEvent: base}
		default:
			s.logger.Debug("Ignoring unknown notification method", "method", method)
			return nil
		}
	}
}

// ============================================================================
// Middleware Implementations (Now methods on Server)
// ============================================================================

func (s *Server) panicRecoveryMiddleware(next HandlerFunc) HandlerFunc {
	return func(ctx context.Context, event Event, logger *slog.Logger) (result any, err *ErrorObject) {
		defer func() {
			if r := recover(); r != nil {
				panicsRecovered.Add(1) // Use global expvar
				stack := string(debug.Stack())
				logger.Error("PANIC recovered in handler", "error", r, "stack", stack)
				err = &ErrorObject{Code: JsonRpcInternalError, Message: fmt.Sprintf("Internal server error from panic: %v", r)}
				result = nil
			}
		}()
		result, err = next(ctx, event, logger)
		return
	}
}

func (s *Server) loggingMiddleware(next HandlerFunc) HandlerFunc {
	return func(ctx context.Context, event Event, logger *slog.Logger) (result any, err *ErrorObject) {
		startTime := time.Now()
		logger.Debug("Handler starting")
		result, err = next(ctx, event, logger)
		duration := time.Since(startTime)
		logLevel := slog.LevelInfo
		if err != nil {
			logLevel = slog.LevelError
		}
		logger.Log(ctx, logLevel, "Handler finished", slog.Duration("duration", duration), slog.Any("error", err))
		return
	}
}

func (s *Server) cancellationMiddleware(next HandlerFunc) HandlerFunc {
	return func(ctx context.Context, event Event, logger *slog.Logger) (result any, err *ErrorObject) {
		reqID := event.GetRequestID()
		if reqID == nil {
			logger.Warn("Cancellation middleware called for notification event type", "type", event.GetType())
			return next(ctx, event, logger)
		}

		reqCtx, cancel := context.WithCancel(ctx)
		s.registerRequest(reqID, cancel, logger)
		defer s.unregisterRequest(reqID, logger)

		result, err = next(reqCtx, event, logger)

		if reqCtx.Err() != nil && err == nil {
			logger.Warn("Request cancelled during handler execution.", "error", reqCtx.Err())
			err = &ErrorObject{Code: JsonRpcRequestCancelled, Message: "Request cancelled"}
			result = nil
		}
		return
	}
}

// ============================================================================
// Request Cancellation Helpers (Now methods on Server)
// ============================================================================

func (s *Server) registerRequest(id any, cancel context.CancelFunc, logger *slog.Logger) {
	if id == nil {
		logger.Warn("Attempted to register request with nil ID for cancellation.")
		return
	}
	if cancel == nil {
		logger.Error("Attempted to register request with nil cancel function", "id", id)
		return
	}
	s.activeRequestMut.Lock()
	defer s.activeRequestMut.Unlock()
	if _, exists := s.activeRequests[id]; exists {
		logger.Warn("Request ID already registered for cancellation, overwriting.", "id", id)
	}
	s.activeRequests[id] = cancel
	logger.Debug("Registered request for cancellation", "id", id)
}

func (s *Server) unregisterRequest(id any, logger *slog.Logger) {
	if id == nil {
		return
	}
	s.activeRequestMut.Lock()
	defer s.activeRequestMut.Unlock()
	delete(s.activeRequests, id)
	logger.Debug("Unregistered request", "id", id)
}

func (s *Server) cancelActiveRequestsForURI(uri DocumentURI, logger *slog.Logger) {
	logger.Warn("Attempting cancellation by URI (currently ineffective)", "uri", uri)
	// TODO: Implement efficient cancellation by URI.
}

// ============================================================================
// Notification Sending Helper (Now method on Server)
// ============================================================================

func (s *Server) sendShowMessageNotification(level MessageType, message string) {
	logger := s.logger
	if s.notificationQueue == nil {
		logger.Error("Cannot send notification, queue not initialized or closed", "level", level, "message", message)
		return
	}
	params := ShowMessageParams{Type: level, Message: message}
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		logger.Error("Failed to marshal ShowMessageParams", "error", err, "message", message)
		return
	}
	notification := NotificationMessage{
		JSONRPC: "2.0",
		Method:  "window/showMessage",
		Params:  paramsJSON,
	}

	select {
	case s.notificationQueue <- notification:
		logger.Debug("Sent showMessage notification to queue", "level", level, "message", message)
	case <-time.After(s.notifySendTimeout):
		notificationQueueTimeout.Add(1) // Use global expvar
		logger.Warn("Timeout sending showMessage notification to queue", "message", message, "timeout", s.notifySendTimeout)
	}
}

// ============================================================================
// Core Handler Logic (Now methods on Server)
// ============================================================================

func (s *Server) handleInitialize(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: initialize")
	var params InitializeParams
	if event.GetRawParams() == nil {
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: "Missing parameters for initialize"}
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Invalid params for initialize", "error", err)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: fmt.Sprintf("Invalid params for initialize: %v", err)}
	}

	if params.ClientInfo != nil {
		logger.Info("Client Info", "name", params.ClientInfo.Name, "version", params.ClientInfo.Version)
	}
	s.clientSupportsSnippets = false
	if params.Capabilities.TextDocument != nil &&
		params.Capabilities.TextDocument.Completion != nil &&
		params.Capabilities.TextDocument.Completion.CompletionItem != nil &&
		params.Capabilities.TextDocument.Completion.CompletionItem.SnippetSupport {
		s.clientSupportsSnippets = true
		logger.Info("Client supports snippets.")
	} else {
		logger.Info("Client does not support snippets.")
	}

	initResult := InitializeResult{
		Capabilities: ServerCapabilities{
			TextDocumentSync:   &TextDocumentSyncOptions{OpenClose: true, Change: TextDocumentSyncKindFull},
			CompletionProvider: &CompletionOptions{},
			HoverProvider:      true,
			DefinitionProvider: true,
		},
		ServerInfo: &ServerInfo{Name: "DeepComplete LSP", Version: s.appVersion},
	}
	return initResult, nil
}

func (s *Server) handleShutdown(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: shutdown")
	return nil, nil
}

func (s *Server) handleCompletion(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: completion")
	var params CompletionParams

	if event.GetRawParams() == nil {
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: "Missing parameters for completion"}
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Failed to unmarshal completion params", "error", err)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: fmt.Sprintf("Invalid params for completion: %v", err)}
	}
	if params.Position.Line < 0 || params.Position.Character < 0 {
		logger.Error("Invalid completion params: negative position", "line", params.Position.Line, "char", params.Position.Character)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: "Invalid position: line and character must be non-negative."}
	}
	absPath, pathErr := ValidateAndGetFilePath(string(params.TextDocument.URI), logger)
	if pathErr != nil {
		logger.Error("Invalid completion params: bad URI", "uri", params.TextDocument.URI, "error", pathErr)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: fmt.Sprintf("Invalid document URI: %v", pathErr)}
	}
	logger = logger.With("uri", string(params.TextDocument.URI), "path", absPath)

	s.docStoreMutex.RLock()
	docInfo, ok := s.documentStore[params.TextDocument.URI]
	s.docStoreMutex.RUnlock() // Release read lock before potential write lock

	if !ok {
		logger.Error("Document not found for completion")
		return nil, nil
	}

	// Update LastAccess time (needs write lock)
	s.docStoreMutex.Lock()
	if docInfoPtr, exists := s.documentStore[params.TextDocument.URI]; exists {
		docInfoPtr.LastAccess = time.Now()
		s.documentStore[params.TextDocument.URI] = docInfoPtr // Update map with modified struct
	}
	s.docStoreMutex.Unlock()

	goLine, goCol, byteOffset, posErr := LspPositionToBytePosition(docInfo.Content, LSPPosition{Line: params.Position.Line, Character: params.Position.Character})
	if posErr != nil {
		logger.Error("Error converting LSP position", "error", posErr)
		return nil, nil
	}

	s.docStoreMutex.RLock()
	currentVersion := docInfo.Version
	s.docStoreMutex.RUnlock()

	logger.Debug("Calling GetCompletionStreamFromFile", "line", goLine, "col", goCol, "version", currentVersion)
	var completionBuf bytes.Buffer
	err := s.completer.GetCompletionStreamFromFile(ctx, absPath, currentVersion, goLine, goCol, &completionBuf)

	if err != nil {
		if errors.Is(err, context.Canceled) {
			logger.Warn("Completion cancelled during core completer execution.")
			return nil, &ErrorObject{Code: JsonRpcRequestCancelled, Message: "Request cancelled"}
		} else if errors.Is(err, ErrOllamaUnavailable) {
			logger.Error("Ollama unavailable for completion", "error", err)
			s.sendShowMessageNotification(MessageTypeError, "Completion backend (Ollama) is unavailable.")
			return nil, &ErrorObject{Code: JsonRpcRequestFailed, Message: "Completion backend unavailable."}
		} else if errors.Is(err, ErrAnalysisFailed) {
			logger.Warn("Analysis failed non-fatally during completion", "error", err)
			s.sendShowMessageNotification(MessageTypeWarning, fmt.Sprintf("Analysis issues: %v. Completion may be less accurate.", err))
		} else {
			logger.Error("Error getting completion from core", "error", err)
			s.sendShowMessageNotification(MessageTypeError, "Internal error generating completion.")
			return nil, &ErrorObject{Code: JsonRpcRequestFailed, Message: "Completion generation failed internally."}
		}
	}

	completionText := completionBuf.String()
	completionResult := CompletionList{IsIncomplete: false, Items: []CompletionItem{}}
	if completionText != "" {
		keepCompletion := true
		if byteOffset >= 0 && byteOffset <= len(docInfo.Content) {
			maxSubsequentLen := 50
			endSubsequent := byteOffset + maxSubsequentLen
			if endSubsequent > len(docInfo.Content) {
				endSubsequent = len(docInfo.Content)
			}
			subsequentBytes := docInfo.Content[byteOffset:endSubsequent]
			subsequentText := string(subsequentBytes)
			if len(subsequentText) > 0 && !match.Match(completionText, subsequentText+"*") {
				logger.Info("Filtering completion due to subsequent text mismatch")
				keepCompletion = false
			}
		} else {
			logger.Warn("Invalid byte offset for filtering", "offset", byteOffset)
		}

		if keepCompletion {
			label := strings.TrimSpace(completionText)
			if firstNewline := strings.Index(label, "\n"); firstNewline != -1 {
				label = label[:firstNewline]
			}
			if len(label) > 50 {
				label = label[:50] + "..."
			}
			// Use package function for kind mapping (defined in lsp_protocol.go)
			kind := mapTypeToCompletionKind(nil) // Pass nil as we don't have the object yet
			logger.Debug("Completion kind mapping not fully implemented, using default.", "default_kind", kind)

			item := CompletionItem{
				Label:  label,
				Kind:   kind,
				Detail: "DeepComplete Suggestion",
			}
			if s.clientSupportsSnippets {
				item.InsertTextFormat = SnippetFormat
				item.InsertText = completionText + "$0"
			} else {
				item.InsertTextFormat = PlainTextFormat
				item.InsertText = completionText
			}
			completionResult.Items = []CompletionItem{item}
		}
	}

	return completionResult, nil
}

func (s *Server) handleHover(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: hover")
	var params HoverParams

	if event.GetRawParams() == nil {
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: "Missing parameters for hover"}
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Failed to unmarshal hover params", "error", err)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: fmt.Sprintf("Invalid params for hover: %v", err)}
	}
	if params.Position.Line < 0 || params.Position.Character < 0 {
		logger.Error("Invalid hover params: negative position", "line", params.Position.Line, "char", params.Position.Character)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: "Invalid position: line and character must be non-negative."}
	}
	absPath, pathErr := ValidateAndGetFilePath(string(params.TextDocument.URI), logger)
	if pathErr != nil {
		logger.Error("Invalid hover params: bad URI", "uri", params.TextDocument.URI, "error", pathErr)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: fmt.Sprintf("Invalid document URI: %v", pathErr)}
	}
	logger = logger.With("uri", string(params.TextDocument.URI), "path", absPath)

	s.docStoreMutex.RLock()
	docInfo, ok := s.documentStore[params.TextDocument.URI]
	s.docStoreMutex.RUnlock() // Release read lock before potential write lock
	if !ok {
		logger.Error("Document not found for hover")
		return nil, nil
	}

	// Update LastAccess time (needs write lock)
	s.docStoreMutex.Lock()
	if docInfoPtr, exists := s.documentStore[params.TextDocument.URI]; exists {
		docInfoPtr.LastAccess = time.Now()
		s.documentStore[params.TextDocument.URI] = docInfoPtr // Update map with modified struct
	}
	s.docStoreMutex.Unlock()

	goLine, goCol, _, posErr := LspPositionToBytePosition(docInfo.Content, LSPPosition{Line: params.Position.Line, Character: params.Position.Character})
	if posErr != nil {
		logger.Error("Error converting LSP position for hover", "error", posErr)
		return nil, nil
	}

	if s.completer == nil || s.completer.GetAnalyzer() == nil {
		logger.Error("Analyzer not available for hover request")
		return nil, nil
	}
	analysisCtx, cancelAnalysis := context.WithTimeout(ctx, s.analysisTimeout)
	defer cancelAnalysis()
	analysisInfo, analysisErr := s.completer.GetAnalyzer().Analyze(analysisCtx, absPath, docInfo.Version, goLine, goCol)

	if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
		logger.Error("Fatal error during analysis/cache check for hover", "error", analysisErr)
		s.sendShowMessageNotification(MessageTypeError, "Failed to analyze code for hover.")
		return nil, &ErrorObject{Code: JsonRpcInternalError, Message: "Failed to analyze code for hover."}
	}
	if analysisErr != nil {
		logger.Warn("Analysis for hover completed with errors", "error", analysisErr)
		s.sendShowMessageNotification(MessageTypeWarning, fmt.Sprintf("Analysis issues during hover: %v", analysisErr))
	}
	if analysisInfo == nil {
		logger.Warn("Analysis for hover returned nil info")
		return nil, nil
	}
	if analysisCtx.Err() != nil {
		logger.Warn("Analysis context cancelled or timed out during hover", "error", analysisCtx.Err())
		return nil, nil
	}

	var hoverResult *HoverResult
	if analysisInfo.IdentifierObject != nil {
		obj := analysisInfo.IdentifierObject
		logger.Debug("Found object for hover", "name", obj.Name(), "type", obj.Type().String())
		// Use the helper function defined in deepcomplete_helpers.go
		hoverContent := formatObjectForHover(obj, analysisInfo, logger)

		if hoverContent != "" {
			var hoverRange *LSPRange
			if analysisInfo.IdentifierAtCursor != nil {
				// Use package function (defined in lsp_protocol.go)
				lspRange, rangeErr := nodeRangeToLSPRange(analysisInfo.TargetFileSet, analysisInfo.IdentifierAtCursor, docInfo.Content, logger)
				if rangeErr != nil {
					logger.Warn("Could not calculate range for hover", "error", rangeErr)
				} else {
					hoverRange = lspRange
				}
			}
			hoverResult = &HoverResult{
				Contents: MarkupContent{
					Kind:  MarkupKindMarkdown,
					Value: hoverContent,
				},
				Range: hoverRange,
			}
		} else {
			logger.Debug("Formatter returned empty content for object", "name", obj.Name())
		}
	} else {
		logger.Debug("No specific identifier object found at cursor for hover")
	}

	return hoverResult, nil
}

func (s *Server) handleDefinition(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: definition")
	var params DefinitionParams

	if event.GetRawParams() == nil {
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: "Missing parameters for definition"}
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Failed to unmarshal definition params", "error", err)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: fmt.Sprintf("Invalid params for definition: %v", err)}
	}
	if params.Position.Line < 0 || params.Position.Character < 0 {
		logger.Error("Invalid definition params: negative position", "line", params.Position.Line, "char", params.Position.Character)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: "Invalid position: line and character must be non-negative."}
	}
	absPath, pathErr := ValidateAndGetFilePath(string(params.TextDocument.URI), logger)
	if pathErr != nil {
		logger.Error("Invalid definition params: bad URI", "uri", params.TextDocument.URI, "error", pathErr)
		return nil, &ErrorObject{Code: JsonRpcInvalidParams, Message: fmt.Sprintf("Invalid document URI: %v", pathErr)}
	}
	logger = logger.With("uri", string(params.TextDocument.URI), "path", absPath)

	s.docStoreMutex.RLock()
	docInfo, ok := s.documentStore[params.TextDocument.URI]
	s.docStoreMutex.RUnlock() // Release read lock before potential write lock
	if !ok {
		logger.Error("Document not found for definition")
		return nil, nil
	}

	// Update LastAccess time (needs write lock)
	s.docStoreMutex.Lock()
	if docInfoPtr, exists := s.documentStore[params.TextDocument.URI]; exists {
		docInfoPtr.LastAccess = time.Now()
		s.documentStore[params.TextDocument.URI] = docInfoPtr // Update map with modified struct
	}
	s.docStoreMutex.Unlock()

	goLine, goCol, _, posErr := LspPositionToBytePosition(docInfo.Content, LSPPosition{Line: params.Position.Line, Character: params.Position.Character})
	if posErr != nil {
		logger.Error("Error converting LSP position for definition", "error", posErr)
		return nil, nil
	}

	if s.completer == nil || s.completer.GetAnalyzer() == nil {
		logger.Error("Analyzer not available for definition request")
		return nil, nil
	}
	analysisCtx, cancelAnalysis := context.WithTimeout(ctx, s.analysisTimeout)
	defer cancelAnalysis()
	analysisInfo, analysisErr := s.completer.GetAnalyzer().Analyze(analysisCtx, absPath, docInfo.Version, goLine, goCol)

	if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
		logger.Error("Fatal error during analysis/cache check for definition", "error", analysisErr)
		s.sendShowMessageNotification(MessageTypeError, "Failed to analyze code for definition.")
		return nil, &ErrorObject{Code: JsonRpcInternalError, Message: "Failed to analyze code for definition."}
	}
	if analysisErr != nil {
		logger.Warn("Analysis for definition completed with errors", "error", analysisErr)
		s.sendShowMessageNotification(MessageTypeWarning, fmt.Sprintf("Analysis issues during definition lookup: %v", analysisErr))
	}
	if analysisInfo == nil {
		logger.Warn("Analysis for definition returned nil info")
		return nil, nil
	}
	if analysisCtx.Err() != nil {
		logger.Warn("Analysis context cancelled or timed out during definition", "error", analysisCtx.Err())
		return nil, nil
	}

	if analysisInfo.IdentifierObject == nil {
		logger.Debug("No identifier object found at cursor for definition")
		return nil, nil
	}

	obj := analysisInfo.IdentifierObject
	defPos := obj.Pos()
	if !defPos.IsValid() {
		logger.Debug("Definition position is invalid for object", "object", obj.Name())
		return nil, nil
	}

	if analysisInfo.TargetFileSet == nil {
		logger.Error("Cannot get definition location: TargetFileSet is nil in analysis info")
		return nil, nil
	}
	fset := analysisInfo.TargetFileSet

	defTokenFile := fset.File(defPos)
	if defTokenFile == nil {
		logger.Error("Cannot get definition location: token.File not found for definition position", "pos", defPos)
		return nil, nil
	}

	defContent, readErr := os.ReadFile(defTokenFile.Name())
	if readErr != nil {
		logger.Error("Failed to read definition file content for position conversion", "file", defTokenFile.Name(), "error", readErr)
		return nil, nil
	}
	// Use package function (defined in lsp_protocol.go)
	defLocation, locErr := tokenPosToLSPLocation(defTokenFile, defPos, defContent, logger)
	if locErr != nil {
		logger.Error("Failed to convert definition position to LSP Location", "error", locErr)
		return nil, nil
	}

	logger.Info("Found definition location", "object", obj.Name(), "location", defLocation)
	return []Location{*defLocation}, nil
}

// ============================================================================
// Notification Handlers (Now methods on Server)
// ============================================================================

func (s *Server) handleInitializedNotification(ctx context.Context, event InitializedNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'initialized' notification.")
}

func (s *Server) handleExitNotification(ctx context.Context, event ExitNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'exit' notification. Signaling shutdown.")
	s.signalShutdown()
}

func (s *Server) handleDidOpenNotification(ctx context.Context, event DidOpenNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'textDocument/didOpen' notification...")
	var params DidOpenTextDocumentParams
	if event.GetRawParams() == nil {
		logger.Error("Missing parameters for didOpen")
		return
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Error decoding didOpen params", "error", err)
		return
	}

	absPath, pathErr := ValidateAndGetFilePath(string(params.TextDocument.URI), logger)
	if pathErr != nil {
		logger.Error("Invalid URI in didOpen", "uri", params.TextDocument.URI, "error", pathErr)
		return
	}
	if params.TextDocument.Version < 0 {
		logger.Error("Invalid negative version in didOpen", "version", params.TextDocument.Version, "uri", params.TextDocument.URI)
		return
	}
	logger = logger.With("uri", string(params.TextDocument.URI), "path", absPath)

	s.docStoreMutex.Lock()
	if len(s.documentStore) >= s.maxOpenDocuments {
		oldestURI := ""
		oldestTime := time.Now()
		for uri, docInfo := range s.documentStore {
			if docInfo.LastAccess.Before(oldestTime) {
				oldestTime = docInfo.LastAccess
				oldestURI = string(uri)
			}
		}
		if oldestURI != "" {
			evictedURI := DocumentURI(oldestURI)
			logger.Info("Max open documents reached, evicting least recently used", "limit", s.maxOpenDocuments, "evicted_uri", evictedURI)
			delete(s.documentStore, evictedURI)
			docsEvicted.Add(1) // Use global expvar
			go s.cancelActiveRequestsForURI(evictedURI, logger)
			if s.completer != nil {
				go s.completer.InvalidateMemoryCacheForURI(string(evictedURI), -1)
			}
		} else {
			logger.Warn("Document store full but failed to find LRU entry to evict.")
		}
	}

	s.documentStore[params.TextDocument.URI] = VersionedContent{
		Content:    []byte(params.TextDocument.Text),
		Version:    params.TextDocument.Version,
		LastAccess: time.Now(),
	}
	s.docStoreMutex.Unlock()

	logger.Info("Opened and stored document", "version", params.TextDocument.Version, "language", params.TextDocument.LanguageID, "length", len(params.TextDocument.Text))
	go s.runAnalysisAndPublishDiagnostics(params.TextDocument.URI, params.TextDocument.Version, logger)
}

func (s *Server) handleDidCloseNotification(ctx context.Context, event DidCloseNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'textDocument/didClose' notification...")
	var params DidCloseTextDocumentParams
	if event.GetRawParams() == nil {
		logger.Error("Missing parameters for didClose")
		return
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Error decoding didClose params", "error", err)
		return
	}

	uri := params.TextDocument.URI
	if _, pathErr := ValidateAndGetFilePath(string(uri), logger); pathErr != nil {
		logger.Error("Invalid URI in didClose", "uri", uri, "error", pathErr)
		return
	}
	logger = logger.With("uri", string(uri))

	s.docStoreMutex.Lock()
	delete(s.documentStore, uri)
	s.docStoreMutex.Unlock()

	logger.Info("Closed and removed document")

	if s.completer != nil {
		uriStr := string(uri)
		go func(uri string) {
			closeLogger := s.logger.With("uri", uri)
			closeLogger.Debug("Attempting memory cache invalidation on didClose")
			if err := s.completer.InvalidateMemoryCacheForURI(uri, -1); err != nil {
				closeLogger.Error("Error invalidating memory cache after didClose", "error", err)
			} else {
				closeLogger.Debug("Memory cache invalidated due to didClose.")
			}
		}(uriStr)
	}

	go s.sendDiagnosticsNotification(uri, nil, logger)
}

func (s *Server) handleDidChangeNotification(ctx context.Context, event DidChangeNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'textDocument/didChange' notification...")
	var params DidChangeTextDocumentParams
	if event.GetRawParams() == nil {
		logger.Error("Missing parameters for didChange")
		return
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Error decoding didChange params", "error", err)
		return
	}

	uri := params.TextDocument.URI
	version := params.TextDocument.Version
	absPath, pathErr := ValidateAndGetFilePath(string(uri), logger)
	if pathErr != nil {
		logger.Error("Invalid URI in didChange", "uri", uri, "error", pathErr)
		return
	}
	if version < 0 {
		logger.Error("Invalid negative version in didChange", "version", version, "uri", uri)
		return
	}
	logger = logger.With("uri", string(uri), "path", absPath, "newVersion", version)

	if len(params.ContentChanges) == 0 {
		logger.Warn("Received didChange notification with no content changes.")
		return
	}
	newText := params.ContentChanges[len(params.ContentChanges)-1].Text

	s.docStoreMutex.Lock()
	existing, ok := s.documentStore[uri]
	shouldUpdate := !ok || version > existing.Version
	if !shouldUpdate {
		if ok {
			logger.Warn("Received out-of-order or redundant document change notification. Ignoring.", "receivedVersion", version, "storedVersion", existing.Version)
		} else {
			logger.Warn("Ignoring didChange notification for unknown reason (shouldUpdate=false, ok=false).", "receivedVersion", version)
		}
		s.docStoreMutex.Unlock()
		return
	}

	s.documentStore[uri] = VersionedContent{
		Content:    []byte(newText),
		Version:    version,
		LastAccess: time.Now(),
	}
	s.docStoreMutex.Unlock()

	logger.Info("Updated document", "newLength", len(newText))

	s.cancelActiveRequestsForURI(uri, logger)

	if s.completer != nil {
		uriStr := string(uri)
		dir := filepath.Dir(absPath)

		go func(d string) {
			bboltLogger := s.logger.With("uri", uriStr, "dir", d)
			bboltLogger.Debug("Attempting bbolt cache invalidation")
			if err := s.completer.InvalidateAnalyzerCache(d); err != nil {
				bboltLogger.Error("Error invalidating bbolt cache after didChange", "error", err)
			} else {
				bboltLogger.Debug("Bbolt cache invalidated due to didChange.")
			}
		}(dir)

		go func(u string, v int) {
			memLogger := s.logger.With("uri", u, "version", v)
			memLogger.Debug("Attempting memory cache invalidation")
			if err := s.completer.InvalidateMemoryCacheForURI(u, v); err != nil {
				memLogger.Error("Error invalidating memory cache after didChange", "error", err)
			} else {
				memLogger.Debug("Memory cache invalidated due to didChange.")
			}
		}(uriStr, version)
	}

	go s.runAnalysisAndPublishDiagnostics(uri, version, logger)
}

func (s *Server) handleDidChangeConfigurationNotification(ctx context.Context, event DidChangeConfigurationNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'workspace/didChangeConfiguration' notification...")
	settingsRaw := event.GetRawParams()
	if settingsRaw == nil {
		logger.Warn("Received didChangeConfiguration with nil settings.")
		return
	}
	logger.Debug("Received configuration change notification", "raw_settings", string(settingsRaw))

	if s.completer == nil {
		logger.Warn("Cannot apply configuration changes, completer not initialized.")
		return
	}

	newConfig := s.completer.GetCurrentConfig()
	settingsGroup := gjson.ParseBytes(settingsRaw)
	configPrefix := "deepcomplete."
	configUpdated := false

	updateIfChanged := func(key string, updateFunc func(val gjson.Result)) {
		result := settingsGroup.Get(configPrefix + key)
		if result.Exists() {
			updateFunc(result)
			configUpdated = true
			logger.Debug("Config key found in notification", "key", configPrefix+key, "value", result.Value())
		}
	}

	updateIfChanged("ollama_url", func(val gjson.Result) { newConfig.OllamaURL = val.String() })
	updateIfChanged("model", func(val gjson.Result) { newConfig.Model = val.String() })
	updateIfChanged("max_tokens", func(val gjson.Result) { newConfig.MaxTokens = int(val.Int()) })
	updateIfChanged("temperature", func(val gjson.Result) { newConfig.Temperature = val.Float() })
	updateIfChanged("log_level", func(val gjson.Result) { newConfig.LogLevel = val.String() })
	updateIfChanged("use_ast", func(val gjson.Result) { newConfig.UseAst = val.Bool() })
	updateIfChanged("use_fim", func(val gjson.Result) { newConfig.UseFim = val.Bool() })
	updateIfChanged("max_preamble_len", func(val gjson.Result) { newConfig.MaxPreambleLen = int(val.Int()) })
	updateIfChanged("max_snippet_len", func(val gjson.Result) { newConfig.MaxSnippetLen = int(val.Int()) })
	updateIfChanged("stop", func(val gjson.Result) {
		if val.IsArray() {
			var stops []string
			for _, item := range val.Array() {
				stops = append(stops, item.String())
			}
			newConfig.Stop = stops
		} else {
			logger.Warn("Config 'stop' value is not an array, ignoring change.", "key", configPrefix+"stop", "value", val.Raw)
		}
	})

	if configUpdated {
		if err := newConfig.Validate(); err != nil {
			logger.Error("Updated configuration is invalid, not applying.", "error", err)
			s.sendShowMessageNotification(MessageTypeError, fmt.Sprintf("Invalid deepcomplete configuration: %v", err))
			return
		}
		if err := s.completer.UpdateConfig(newConfig); err != nil {
			logger.Error("Error applying updated configuration", "error", err)
			s.sendShowMessageNotification(MessageTypeError, fmt.Sprintf("Failed to apply configuration: %v", err))
		} else {
			logger.Info("Successfully applied updated configuration from client.")
			if settingsGroup.Get(configPrefix + "log_level").Exists() {
				s.sendShowMessageNotification(MessageTypeInfo, "Log level configuration changed. Restart LSP server for change to take full effect.")
			}
		}
	} else {
		logger.Info("No relevant configuration changes detected in notification.")
	}
}

func (s *Server) handleCancelRequestNotification(ctx context.Context, event CancelRequestEvent, logger *slog.Logger) {
	logger.Info("Handling '$/cancelRequest' notification...")
	if event.GetRawParams() == nil {
		logger.Warn("Received cancel request with nil params.")
		return
	}
	idResult := gjson.GetBytes(event.GetRawParams(), "id")
	if !idResult.Exists() {
		logger.Warn("Received cancel request without ID.")
		return
	}
	reqID := idResult.Value()

	logger = logger.With("cancelledRequestID", reqID)
	logger.Info("Received cancellation request")

	s.activeRequestMut.Lock()
	cancelFunc, ok := s.activeRequests[reqID]
	delete(s.activeRequests, reqID)
	s.activeRequestMut.Unlock()

	if ok && cancelFunc != nil {
		cancelFunc()
		logger.Info("Cancelled active request")
	} else {
		logger.Warn("No active request found to cancel (already finished or invalid ID)")
	}
}

// ============================================================================
// Diagnostic Helpers (Now methods on Server)
// ============================================================================

func (s *Server) runAnalysisAndPublishDiagnostics(uri DocumentURI, version int, logger *slog.Logger) {
	logger = logger.With("uri", string(uri), "version", version)
	logger.Info("Starting background analysis for diagnostics...")
	startTime := time.Now()

	s.docStoreMutex.RLock()
	docInfo, ok := s.documentStore[uri]
	s.docStoreMutex.RUnlock()
	if !ok {
		logger.Warn("Document not found in store for diagnostic analysis.")
		return
	}
	absPath, pathErr := ValidateAndGetFilePath(string(uri), logger)
	if pathErr != nil {
		logger.Error("Invalid URI for diagnostic analysis", "error", pathErr)
		return
	}

	if s.completer == nil || s.completer.GetAnalyzer() == nil {
		logger.Error("Analyzer not available for diagnostic analysis")
		return
	}

	analysisCtx, cancelAnalysis := context.WithTimeout(context.Background(), s.analysisTimeout)
	defer cancelAnalysis()

	analysisInfo, analysisErr := s.completer.GetAnalyzer().Analyze(analysisCtx, absPath, version, 1, 1)

	duration := time.Since(startTime)
	if analysisErr != nil {
		if errors.Is(analysisErr, ErrAnalysisFailed) {
			logger.Warn("Analysis for diagnostics completed with non-fatal errors", "duration", duration, "error", analysisErr)
		} else {
			logger.Error("Fatal error during analysis for diagnostics", "duration", duration, "error", analysisErr)
			s.sendDiagnosticsNotification(uri, nil, logger)
			return
		}
	} else {
		logger.Info("Analysis for diagnostics completed successfully", "duration", duration)
	}

	var lspDiagnostics []LspDiagnostic
	if analysisInfo != nil && len(analysisInfo.Diagnostics) > 0 {
		lspDiagnostics = s.convertInternalDiagnosticsToLSP(analysisInfo.Diagnostics, docInfo.Content, logger)
		logger.Debug("Converted internal diagnostics to LSP format", "count", len(lspDiagnostics))
	} else {
		logger.Debug("No internal diagnostics found to convert.")
		lspDiagnostics = []LspDiagnostic{}
	}

	s.sendDiagnosticsNotification(uri, lspDiagnostics, logger)
}

func (s *Server) convertInternalDiagnosticsToLSP(internalDiags []Diagnostic, content []byte, logger *slog.Logger) []LspDiagnostic {
	lspDiags := make([]LspDiagnostic, 0, len(internalDiags))
	for _, intDiag := range internalDiags {
		// Assuming intDiag.Range.Start/End.Character holds byte offset
		// Use function defined in lsp_protocol.go
		startLine, startChar, startErr := byteOffsetToLSPPosition(content, intDiag.Range.Start.Character, logger)
		endLine, endChar, endErr := byteOffsetToLSPPosition(content, intDiag.Range.End.Character, logger)

		if startErr != nil || endErr != nil {
			logger.Warn("Failed to convert diagnostic range offsets to LSP positions, skipping diagnostic.",
				"message", intDiag.Message, "start_offset", intDiag.Range.Start.Character, "end_offset", intDiag.Range.End.Character, "start_err", startErr, "end_err", endErr)
			continue
		}

		var lspSeverity LspDiagnosticSeverity
		switch intDiag.Severity {
		case SeverityError:
			lspSeverity = LspSeverityError
		case SeverityWarning:
			lspSeverity = LspSeverityWarning
		case SeverityInfo:
			lspSeverity = LspSeverityInfo
		case SeverityHint:
			lspSeverity = LspSeverityHint
		default:
			lspSeverity = LspSeverityWarning
		}

		lspDiags = append(lspDiags, LspDiagnostic{
			Range: LSPRange{
				Start: LSPPosition{Line: startLine, Character: startChar},
				End:   LSPPosition{Line: endLine, Character: endChar},
			},
			Severity: lspSeverity,
			Code:     intDiag.Code,
			Source:   intDiag.Source,
			Message:  intDiag.Message,
		})
	}
	return lspDiags
}

func (s *Server) sendDiagnosticsNotification(uri DocumentURI, diagnostics []LspDiagnostic, logger *slog.Logger) {
	if s.notificationQueue == nil {
		logger.Error("Cannot send diagnostics notification, queue not initialized or closed", "uri", uri)
		return
	}
	if diagnostics == nil {
		diagnostics = []LspDiagnostic{}
	}

	params := PublishDiagnosticsParams{
		URI:         uri,
		Diagnostics: diagnostics,
	}
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		logger.Error("Failed to marshal PublishDiagnosticsParams", "error", err, "uri", uri)
		return
	}
	notification := NotificationMessage{
		JSONRPC: "2.0",
		Method:  "textDocument/publishDiagnostics",
		Params:  paramsJSON,
	}

	select {
	case s.notificationQueue <- notification:
		diagnosticsPublished.Add(1) // Use global expvar
		logger.Info("Sent diagnostics notification to queue", "uri", uri, "count", len(diagnostics))
	case <-time.After(s.notifySendTimeout): // Use server config
		notificationQueueTimeout.Add(1) // Use global expvar
		logger.Warn("Timeout sending diagnostics notification to queue", "uri", uri, "timeout", s.notifySendTimeout)
	}
}
