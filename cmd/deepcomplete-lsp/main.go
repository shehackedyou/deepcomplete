package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"expvar" // Cycle 3: Added expvar
	"fmt"
	"io"
	"log"              // Still used for initial fatal errors before slog is set up
	"log/slog"         // Cycle 3: Added slog
	"net/http"         // Cycle 3: Added http
	_ "net/http/pprof" // Cycle 3: Added pprof endpoint registration
	"net/url"
	"os"
	"path/filepath"
	"runtime"       // Cycle 3: Added runtime
	"runtime/debug" // Keep for stack traces
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson" // Cycle 2: Added gjson dependency
	"github.com/tidwall/match" // Cycle 3: Added match dependency

	// NOTE: Replace with your actual module path
	"github.com/shehackedyou/deepcomplete"
)

// ============================================================================
// Global Variables & Constants
// ============================================================================

// VersionedContent stores document content along with its version.
type VersionedContent struct {
	Content []byte
	Version int
}

var (
	// completer is initialized in main and used by handlers.
	completer *deepcomplete.DeepCompleter

	// documentStore holds the content and version of open documents, keyed by URI.
	// Cycle 2: Changed value type to VersionedContent
	documentStore = make(map[DocumentURI]VersionedContent)
	docStoreMutex sync.RWMutex // Protects concurrent access to documentStore.

	// clientSupportsSnippets tracks if the connected client supports snippet format. (Set during initialize).
	clientSupportsSnippets bool

	// --- Event-Driven Architecture Components ---

	// eventQueue holds incoming LSP messages parsed into Event structs.
	eventQueue chan Event
	// responseQueue holds results/errors from processed requests waiting to be written.
	responseQueue chan ResponseWorkItem
	// shutdownChan signals graceful shutdown.
	shutdownChan chan struct{}
	// wg tracks active goroutines for graceful shutdown.
	wg sync.WaitGroup

	// activeRequests tracks ongoing requests that can be cancelled. Maps request ID to context cancel func.
	activeRequests   = make(map[any]context.CancelFunc)
	activeRequestMut sync.Mutex // Protects activeRequests map.

	// Default buffer size for channels.
	defaultQueueSize = 100

	// Cycle 3: Monitoring variables
	eventsReceived  = expvar.NewInt("lsp.eventsReceived")
	eventsIgnored   = expvar.NewInt("lsp.eventsIgnored")
	requestsSpawned = expvar.NewInt("lsp.requestsSpawned")
	responsesSent   = expvar.NewInt("lsp.responsesSent")
	errorsReported  = expvar.NewInt("lsp.errorsReported")
	panicsRecovered = expvar.NewInt("lsp.panicsRecovered")
)

// JSON-RPC Standard Error Codes
const (
	ParseError     int = -32700
	InvalidRequest int = -32600
	MethodNotFound int = -32601
	InvalidParams  int = -32602
	InternalError  int = -32603
	// Server specific codes
	RequestFailed    int = -32000 // General request failure
	RequestCancelled int = -32800 // LSP standard for cancelled request
)

// ============================================================================
// Event Definitions
// ============================================================================

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

func (e BaseEvent) GetType() string {
	return e.Type
}
func (e BaseEvent) GetRequestID() any {
	return e.RequestID
}
func (e BaseEvent) GetRawParams() json.RawMessage {
	return e.RawParams
}

// InitializeRequestEvent represents the 'initialize' request.
type InitializeRequestEvent struct {
	BaseEvent
	// Params are parsed on demand from RawParams
}

// ShutdownRequestEvent represents the 'shutdown' request.
type ShutdownRequestEvent struct {
	BaseEvent
}

// CompletionRequestEvent represents the 'textDocument/completion' request.
type CompletionRequestEvent struct {
	BaseEvent
	// Params are parsed on demand from RawParams
}

// DidOpenNotificationEvent represents the 'textDocument/didOpen' notification.
type DidOpenNotificationEvent struct {
	BaseEvent
	// Params are parsed on demand from RawParams
}

// DidCloseNotificationEvent represents the 'textDocument/didClose' notification.
type DidCloseNotificationEvent struct {
	BaseEvent
	// Params are parsed on demand from RawParams
}

// DidChangeNotificationEvent represents the 'textDocument/didChange' notification.
type DidChangeNotificationEvent struct {
	BaseEvent
	// Params are parsed on demand from RawParams
}

// DidChangeConfigurationNotificationEvent represents the 'workspace/didChangeConfiguration' notification.
type DidChangeConfigurationNotificationEvent struct {
	BaseEvent
	// Params are parsed on demand from RawParams
}

// InitializedNotificationEvent represents the 'initialized' notification.
type InitializedNotificationEvent struct {
	BaseEvent
}

// ExitNotificationEvent represents the 'exit' notification.
type ExitNotificationEvent struct {
	BaseEvent
}

// CancelRequestEvent represents the '$/cancelRequest' notification.
type CancelRequestEvent struct {
	BaseEvent
	// Params are parsed on demand from RawParams
}

// UnknownEvent represents an unhandled or unparseable message.
type UnknownEvent struct {
	BaseEvent
	ParseError error
}

// ResponseWorkItem holds the result or error for a completed request, ready to be sent.
type ResponseWorkItem struct {
	RequestID any          // Original request ID.
	Result    any          `json:"result,omitempty"`
	Error     *ErrorObject `json:"error,omitempty"`
}

// ============================================================================
// JSON-RPC Structures
// ============================================================================

// RequestMessage used only for initial parsing to get ID/Method
type RequestMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type ResponseMessage struct {
	JSONRPC string       `json:"jsonrpc"`
	ID      any          `json:"id,omitempty"`
	Result  any          `json:"result,omitempty"`
	Error   *ErrorObject `json:"error,omitempty"`
}

// NotificationMessage used only for initial parsing to get Method
type NotificationMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type ErrorObject struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// ============================================================================
// LSP Specific Structures (Simplified Local Definitions)
// ============================================================================
// These remain the same, used for unmarshalling specific params when needed.

type DocumentURI string
type Position struct {
	Line      uint32 `json:"line"`
	Character uint32 `json:"character"`
}
type Range struct {
	Start Position `json:"start"`
	End   Position `json:"end"`
}
type Location struct {
	URI   DocumentURI `json:"uri"`
	Range Range       `json:"range"`
}
type TextDocumentIdentifier struct {
	URI DocumentURI `json:"uri"`
}
type TextDocumentItem struct {
	URI        DocumentURI `json:"uri"`
	LanguageID string      `json:"languageId"`
	Version    int         `json:"version"`
	Text       string      `json:"text"`
}
type InitializeParams struct {
	ProcessID             int                `json:"processId,omitempty"`
	RootURI               DocumentURI        `json:"rootUri,omitempty"`
	ClientInfo            *ClientInfo        `json:"clientInfo,omitempty"`
	Capabilities          ClientCapabilities `json:"capabilities"`
	InitializationOptions json.RawMessage    `json:"initializationOptions,omitempty"`
}
type ClientInfo struct {
	Name    string `json:"name,omitempty"`
	Version string `json:"version,omitempty"`
}
type ClientCapabilities struct {
	Workspace    *WorkspaceClientCapabilities    `json:"workspace,omitempty"`
	TextDocument *TextDocumentClientCapabilities `json:"textDocument,omitempty"`
}
type WorkspaceClientCapabilities struct {
	Configuration bool `json:"configuration,omitempty"`
}
type TextDocumentClientCapabilities struct {
	Completion *CompletionClientCapabilities `json:"completion,omitempty"`
}
type CompletionClientCapabilities struct {
	CompletionItem *CompletionItemClientCapabilities `json:"completionItem,omitempty"`
}
type CompletionItemClientCapabilities struct {
	SnippetSupport bool `json:"snippetSupport,omitempty"`
}
type InitializeResult struct {
	Capabilities ServerCapabilities `json:"capabilities"`
	ServerInfo   *ServerInfo        `json:"serverInfo,omitempty"`
}
type ServerCapabilities struct {
	TextDocumentSync   *TextDocumentSyncOptions `json:"textDocumentSync,omitempty"`
	CompletionProvider *CompletionOptions       `json:"completionProvider,omitempty"`
}
type TextDocumentSyncOptions struct {
	OpenClose bool                 `json:"openClose,omitempty"`
	Change    TextDocumentSyncKind `json:"change,omitempty"`
}
type TextDocumentSyncKind int

const (
	TextDocumentSyncKindNone TextDocumentSyncKind = 0
	TextDocumentSyncKindFull TextDocumentSyncKind = 1
)

type CompletionOptions struct{}
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
}
type DidOpenTextDocumentParams struct {
	TextDocument TextDocumentItem `json:"textDocument"`
}
type DidCloseTextDocumentParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}
type DidChangeTextDocumentParams struct {
	TextDocument   VersionedTextDocumentIdentifier  `json:"textDocument"`
	ContentChanges []TextDocumentContentChangeEvent `json:"contentChanges"`
}
type VersionedTextDocumentIdentifier struct {
	TextDocumentIdentifier
	Version int `json:"version"`
}
type TextDocumentContentChangeEvent struct {
	Text string `json:"text"`
}
type DidChangeConfigurationParams struct {
	Settings json.RawMessage `json:"settings"`
}
type CompletionParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     Position               `json:"position"`
	Context      *CompletionContext     `json:"context,omitempty"`
}
type CompletionContext struct {
	TriggerKind      CompletionTriggerKind `json:"triggerKind"`
	TriggerCharacter string                `json:"triggerCharacter,omitempty"`
}
type CompletionTriggerKind int

const (
	CompletionTriggerKindInvoked              CompletionTriggerKind = 1
	CompletionTriggerKindTriggerChar          CompletionTriggerKind = 2
	CompletionTriggerKindTriggerForIncomplete CompletionTriggerKind = 3
)

type CompletionList struct {
	IsIncomplete bool             `json:"isIncomplete"`
	Items        []CompletionItem `json:"items"`
}
type CompletionItem struct {
	Label            string             `json:"label"`
	Kind             CompletionItemKind `json:"kind,omitempty"`
	Detail           string             `json:"detail,omitempty"`
	Documentation    string             `json:"documentation,omitempty"`
	InsertTextFormat InsertTextFormat   `json:"insertTextFormat,omitempty"`
	InsertText       string             `json:"insertText,omitempty"`
}
type CompletionItemKind int

const (
	CompletionItemKindText     CompletionItemKind = 1
	CompletionItemKindFunction CompletionItemKind = 3
	CompletionItemKindVariable CompletionItemKind = 6
	CompletionItemKindKeyword  CompletionItemKind = 14
)

type InsertTextFormat int

const (
	PlainTextFormat InsertTextFormat = 1
	SnippetFormat   InsertTextFormat = 2
)

type CancelParams struct {
	ID any `json:"id"`
}

// ============================================================================
// Main Server Logic & Event Loop
// ============================================================================

func main() {
	// Setup logging *before* initializing slog
	logFile, err := os.OpenFile("deepcomplete-lsp.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0660)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err) // Use standard log for initial fatal error
	}
	defer logFile.Close()

	// Cycle 3: Initialize slog
	logLevel := slog.LevelDebug // Default level, could be configurable via flag/env
	logWriter := io.MultiWriter(os.Stderr, logFile)
	// Add source location to logs for easier debugging
	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: true}
	// Using TextHandler for more human-readable logs during dev, JSONHandler might be better for production aggregation
	handler := slog.NewTextHandler(logWriter, &handlerOpts)
	// handler := slog.NewJSONHandler(logWriter, &handlerOpts)
	logger := slog.New(handler)
	slog.SetDefault(logger) // Set as default for convenience

	slog.Info("DeepComplete LSP server starting...")

	// Cycle 3: Enable profiling rates
	runtime.SetBlockProfileRate(1)     // Enable blocking profile (1 = report all blocking events)
	runtime.SetMutexProfileFraction(1) // Enable mutex profile (1 = report all contended mutexes)
	slog.Info("Enabled block and mutex profiling")

	// Cycle 3: Start pprof/expvar HTTP server
	// Note: In production, bind to localhost or protect this endpoint
	debugListenAddr := "localhost:6061" // TODO: Make configurable?
	go func() {
		slog.Info("Starting debug server for pprof/expvar", "addr", debugListenAddr)
		// pprof endpoints are registered by importing net/http/pprof
		// expvar endpoint is registered by importing expvar at /debug/vars
		if err := http.ListenAndServe(debugListenAddr, nil); err != nil {
			slog.Error("Debug server failed", "error", err)
		}
	}()

	// Cycle 3: Publish expvar metrics for queue lengths
	expvar.Publish("eventQueueLength", expvar.Func(func() any {
		// Check if queue is initialized before accessing length
		if eventQueue == nil {
			return 0
		}
		return len(eventQueue)
	}))
	expvar.Publish("responseQueueLength", expvar.Func(func() any {
		if responseQueue == nil {
			return 0
		}
		return len(responseQueue)
	}))
	// Other counters are incremented directly where events occur

	// Initialize core service
	var initErr error
	completer, initErr = deepcomplete.NewDeepCompleter() // Assumes NewDeepCompleter uses slog now
	if initErr != nil {
		slog.Error("Failed to initialize DeepCompleter service", "error", initErr)
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing DeepCompleter service...")
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()
	slog.Info("DeepComplete service initialized.")

	// Initialize queues and shutdown channel
	eventQueue = make(chan Event, defaultQueueSize)
	responseQueue = make(chan ResponseWorkItem, defaultQueueSize)
	shutdownChan = make(chan struct{})
	slog.Info("Initialized event and response queues", "size", defaultQueueSize)

	// Start goroutines
	wg.Add(3) // Reader, Dispatcher, Writer
	go runReader(os.Stdin)
	go runDispatcher()
	go runResponseWriter(os.Stdout)
	slog.Info("Started Reader, Dispatcher, and Writer goroutines.")

	// Wait for shutdown signal
	<-shutdownChan
	slog.Info("Shutdown signal received.")

	// Wait for goroutines to finish
	slog.Info("Waiting for goroutines to finish...")
	wg.Wait()
	slog.Info("All goroutines finished. Exiting.")
}

// runReader reads messages from the client, parses them, and sends them to the eventQueue.
func runReader(r io.Reader) {
	defer wg.Done()
	defer func() {
		if r := recover(); r != nil {
			panicsRecovered.Add(1)
			slog.Error("PANIC in runReader", "error", r, "stack", string(debug.Stack()))
		}
		slog.Info("Reader goroutine stopped.")
	}()

	reader := bufio.NewReader(r)
	for {
		select {
		case <-shutdownChan:
			return // Stop reading on shutdown signal
		default:
			content, err := readMessage(reader)
			if err != nil {
				if errors.Is(err, io.EOF) {
					slog.Info("Client closed connection (EOF). Stopping reader.")
					// Signal shutdown gracefully if not already shutting down
					select {
					case <-shutdownChan:
					default:
						close(shutdownChan)
					}
					return
				}
				slog.Error("Error reading message", "error", err)
				time.Sleep(100 * time.Millisecond) // Avoid busy-loop on persistent read errors
				continue
			}

			eventsReceived.Add(1)
			slog.Debug("Received message content", "content", string(content))
			event := parseMessageToEvent(content)
			if event == nil {
				eventsIgnored.Add(1)
				continue // Ignore events parsed as nil (e.g., unknown notifications)
			}
			slog.Debug("Parsed event", "type", fmt.Sprintf("%T", event), "requestID", event.GetRequestID())

			// Send event to queue, handle potential shutdown during send
			select {
			case eventQueue <- event:
				// Event sent successfully
			case <-shutdownChan:
				slog.Warn("Shutdown signaled while sending event to queue.")
				return
			case <-time.After(5 * time.Second): // Timeout for sending to queue
				slog.Error("Timeout sending event to queue (possible deadlock or slow dispatcher?)", "eventType", event.GetType())
				// Depending on severity, might want to signal shutdown or drop event
			}
		}
	}
}

// parseMessageToEvent parses raw message content into a specific Event struct.
// Uses gjson (Cycle 2) to determine type without full unmarshalling initially.
func parseMessageToEvent(content []byte) Event {
	// Fix 1: Convert gjson Raw string to json.RawMessage ([]byte)
	rawParams := json.RawMessage(gjson.GetBytes(content, "params").Raw)
	base := BaseEvent{RawParams: rawParams}

	// Check if it's a request (has "id") or notification
	idResult := gjson.GetBytes(content, "id")
	methodResult := gjson.GetBytes(content, "method")
	if !methodResult.Exists() {
		slog.Error("Received message without method", "content", string(content))
		return UnknownEvent{BaseEvent: base, ParseError: errors.New("message missing method field")}
	}
	base.Type = methodResult.String()

	if idResult.Exists() {
		// It's a request
		base.RequestID = idResult.Value() // Store the ID (number or string)
		switch base.Type {
		case "initialize":
			return InitializeRequestEvent{BaseEvent: base}
		case "shutdown":
			return ShutdownRequestEvent{BaseEvent: base}
		case "textDocument/completion":
			return CompletionRequestEvent{BaseEvent: base}
		default:
			slog.Warn("Received unknown request method", "method", base.Type, "requestID", base.RequestID)
			return UnknownEvent{BaseEvent: base, ParseError: fmt.Errorf("unknown request method: %s", base.Type)}
		}
	} else {
		// It's a notification
		base.RequestID = nil
		switch base.Type {
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
			slog.Debug("Ignoring unknown notification method", "method", base.Type)
			// Don't treat unknown notifications as errors, just ignore them per LSP spec
			return nil // Return nil to indicate it should be ignored
		}
	}
}

// runDispatcher reads events from eventQueue and dispatches them to handlers or workers.
func runDispatcher() {
	defer wg.Done()
	defer func() {
		if r := recover(); r != nil {
			panicsRecovered.Add(1)
			slog.Error("PANIC in runDispatcher", "error", r, "stack", string(debug.Stack()))
		}
		slog.Info("Dispatcher goroutine stopped.")
	}()

	// Create a main context for the server lifecycle
	serverCtx, cancelServer := context.WithCancel(context.Background())
	defer cancelServer() // Ensure cancellation propagates on dispatcher exit

	for {
		select {
		case <-shutdownChan:
			slog.Info("Dispatcher received shutdown signal.")
			cancelServer()                     // Cancel context for all active workers
			time.Sleep(200 * time.Millisecond) // Wait briefly
			return
		case event, ok := <-eventQueue:
			if !ok {
				slog.Info("Event queue closed. Stopping dispatcher.")
				cancelServer()
				return // Exit if the queue is closed
			}

			// Cycle 3: Create contextual logger
			eventLogger := slog.Default().With(
				"requestID", event.GetRequestID(), // Will be nil for notifications
				"eventType", fmt.Sprintf("%T", event),
			)
			eventLogger.Info("Dispatcher processing event")

			// --- Basic Middleware Concept Placeholder ---
			// --- End Middleware Placeholder ---

			switch ev := event.(type) {
			// --- Requests (Spawn Worker) ---
			case InitializeRequestEvent:
				requestsSpawned.Add(1)
				// Pass contextual logger to worker
				go handleInitializeRequest(serverCtx, ev, eventLogger.With("handler", "Initialize"))
			case ShutdownRequestEvent:
				requestsSpawned.Add(1)
				go handleShutdownRequest(serverCtx, ev, eventLogger.With("handler", "Shutdown"))
			case CompletionRequestEvent:
				requestsSpawned.Add(1)
				go handleCompletionRequest(serverCtx, ev, eventLogger.With("handler", "Completion"))

			// --- Notifications (Handle Directly or Quick Worker) ---
			case InitializedNotificationEvent:
				handleInitializedNotification(serverCtx, ev, eventLogger.With("handler", "Initialized"))
			case ExitNotificationEvent:
				handleExitNotification(serverCtx, ev, eventLogger.With("handler", "Exit")) // Triggers shutdown
			case DidOpenNotificationEvent:
				handleDidOpenNotification(serverCtx, ev, eventLogger.With("handler", "DidOpen"))
			case DidCloseNotificationEvent:
				handleDidCloseNotification(serverCtx, ev, eventLogger.With("handler", "DidClose"))
			case DidChangeNotificationEvent:
				handleDidChangeNotification(serverCtx, ev, eventLogger.With("handler", "DidChange"))
			case DidChangeConfigurationNotificationEvent:
				handleDidChangeConfigurationNotification(serverCtx, ev, eventLogger.With("handler", "DidChangeConfiguration"))
			case CancelRequestEvent:
				handleCancelRequestNotification(serverCtx, ev, eventLogger.With("handler", "CancelRequest"))

			// --- Error/Unknown ---
			case UnknownEvent:
				eventLogger.Warn("Dispatcher received UnknownEvent", "error", ev.ParseError)
				if ev.GetRequestID() != nil {
					errorsReported.Add(1)
					// Send error response for unknown requests
					responseQueue <- ResponseWorkItem{
						RequestID: ev.GetRequestID(),
						Error:     &ErrorObject{Code: MethodNotFound, Message: fmt.Sprintf("Method not found: %s", ev.GetType())},
					}
				}
			default:
				eventLogger.Warn("Dispatcher received unhandled event type")
			}
		}
	}
}

// runResponseWriter reads completed work items and writes responses to the client.
func runResponseWriter(w io.Writer) {
	defer wg.Done()
	defer func() {
		if r := recover(); r != nil {
			panicsRecovered.Add(1)
			slog.Error("PANIC in runResponseWriter", "error", r, "stack", string(debug.Stack()))
		}
		slog.Info("Response Writer goroutine stopped.")
	}()

	// Use a mutex for writing to stdout, although it's often safe for concurrent writes.
	var writerMu sync.Mutex

	for {
		select {
		case <-shutdownChan:
			slog.Info("Response Writer received shutdown signal.")
			// Process any remaining items in the queue? Optional.
			for len(responseQueue) > 0 {
				workItem := <-responseQueue
				slog.Debug("Processing remaining response item after shutdown", "requestID", workItem.RequestID)
				response := ResponseMessage{JSONRPC: "2.0", ID: workItem.RequestID, Result: workItem.Result, Error: workItem.Error}
				writerMu.Lock()
				_ = writeMessage(w, response) // Ignore write errors during shutdown flush
				writerMu.Unlock()
			}
			return
		case workItem, ok := <-responseQueue:
			if !ok {
				slog.Info("Response queue closed. Stopping writer.")
				return
			}

			response := ResponseMessage{
				JSONRPC: "2.0",
				ID:      workItem.RequestID,
				Result:  workItem.Result,
				Error:   workItem.Error,
			}

			writerMu.Lock()
			err := writeMessage(w, response)
			writerMu.Unlock()

			if err != nil {
				slog.Error("Error writing response", "requestID", workItem.RequestID, "error", err)
				// Consider implications if writing fails (client disconnected?)
			} else {
				responsesSent.Add(1)
				slog.Debug("Successfully wrote response", "requestID", workItem.RequestID)
			}
		}
	}
}

// ============================================================================
// Event Handlers (Executed by Dispatcher or Workers)
// ============================================================================

// --- Request Handlers (Run in separate goroutines) ---

// Cycle 3: Added contextual logger argument
func handleInitializeRequest(ctx context.Context, event InitializeRequestEvent, logger *slog.Logger) {
	// Register request for potential cancellation
	reqCtx, cancel := context.WithCancel(ctx)
	registerRequest(event.GetRequestID(), cancel, logger) // Pass logger
	defer unregisterRequest(event.GetRequestID(), logger) // Pass logger

	logger.Info("Worker handling 'initialize' request...")
	var params InitializeParams
	var result InitializeResult
	var errObj *ErrorObject

	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Invalid params for initialize", "error", err)
		errObj = &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid params for initialize: %v", err)}
	} else {
		if params.ClientInfo != nil {
			logger.Info("Client Info", "name", params.ClientInfo.Name, "version", params.ClientInfo.Version)
		}
		// Check client capabilities
		clientSupportsSnippets = false // Default
		if params.Capabilities.TextDocument != nil &&
			params.Capabilities.TextDocument.Completion != nil &&
			params.Capabilities.TextDocument.Completion.CompletionItem != nil &&
			params.Capabilities.TextDocument.Completion.CompletionItem.SnippetSupport {
			clientSupportsSnippets = true
			logger.Info("Client supports snippets.")
		} else {
			logger.Info("Client does not support snippets.")
		}
		// Advertise server capabilities.
		result = InitializeResult{
			Capabilities: ServerCapabilities{
				TextDocumentSync:   &TextDocumentSyncOptions{OpenClose: true, Change: TextDocumentSyncKindFull},
				CompletionProvider: &CompletionOptions{},
			},
			ServerInfo: &ServerInfo{Name: "DeepComplete LSP", Version: "0.0.1" /* TODO: Get version dynamically */},
		}
	}

	// Send result/error back to the writer goroutine
	select {
	case responseQueue <- ResponseWorkItem{RequestID: event.GetRequestID(), Result: result, Error: errObj}:
		if errObj != nil {
			errorsReported.Add(1)
		}
	case <-reqCtx.Done(): // Check if the request was cancelled *during* processing
		logger.Warn("Initialize request cancelled during processing.")
		errorsReported.Add(1)
		responseQueue <- ResponseWorkItem{
			RequestID: event.GetRequestID(),
			Error:     &ErrorObject{Code: RequestCancelled, Message: "Request cancelled"},
		}
	case <-shutdownChan:
		logger.Warn("Shutdown occurred while sending initialize response.")
	}
}

// Cycle 3: Added contextual logger argument
func handleShutdownRequest(ctx context.Context, event ShutdownRequestEvent, logger *slog.Logger) {
	// Register request for potential cancellation
	reqCtx, cancel := context.WithCancel(ctx)
	registerRequest(event.GetRequestID(), cancel, logger)
	defer unregisterRequest(event.GetRequestID(), logger)

	logger.Info("Worker handling 'shutdown' request...")

	// Signal successful shutdown preparation
	select {
	case responseQueue <- ResponseWorkItem{RequestID: event.GetRequestID(), Result: nil, Error: nil}:
		logger.Info("Sent successful shutdown response. Server will exit on 'exit' notification.")
	case <-reqCtx.Done():
		logger.Warn("Shutdown request cancelled.")
		errorsReported.Add(1)
		responseQueue <- ResponseWorkItem{
			RequestID: event.GetRequestID(),
			Error:     &ErrorObject{Code: RequestCancelled, Message: "Shutdown request cancelled"},
		}
	case <-shutdownChan:
		logger.Warn("Shutdown occurred while sending shutdown response.")
	}
}

// Cycle 3: Added contextual logger argument
func handleCompletionRequest(ctx context.Context, event CompletionRequestEvent, logger *slog.Logger) {
	// Register request for potential cancellation
	reqCtx, cancel := context.WithCancel(ctx)
	registerRequest(event.GetRequestID(), cancel, logger)
	defer unregisterRequest(event.GetRequestID(), logger) // Cleanup cancellation registration

	logger.Info("Worker handling 'textDocument/completion' request...")
	var result CompletionList
	var errObj *ErrorObject
	var params CompletionParams // Declare params here

	// Use goto label for cleaner exit path on error/cancellation
	defer func() {
		// This deferred function sends the final response
		select {
		case responseQueue <- ResponseWorkItem{RequestID: event.GetRequestID(), Result: result, Error: errObj}:
			if errObj != nil {
				errorsReported.Add(1)
			}
		case <-ctx.Done(): // Use original server context for checking server shutdown during send
			logger.Warn("Shutdown occurred while sending completion response.")
		case <-reqCtx.Done(): // Check request context again in case cancelled *after* processing but *before* send
			logger.Warn("Completion request cancelled just before sending response.")
			// Avoid sending duplicate cancellation error if already set
			if errObj == nil || errObj.Code != RequestCancelled {
				errorsReported.Add(1)
				// Ensure responseQueue is still writable, otherwise log
				select {
				case responseQueue <- ResponseWorkItem{
					RequestID: event.GetRequestID(),
					Error:     &ErrorObject{Code: RequestCancelled, Message: "Request cancelled"},
				}:
				default:
					logger.Error("Response queue likely closed, could not send cancellation error for completion.")
				}
			}
		}
	}()

	// Cycle 2: Use gjson to extract specific fields first
	paramsRaw := event.GetRawParams()
	uriResult := gjson.GetBytes(paramsRaw, "textDocument.uri")
	posResult := gjson.GetBytes(paramsRaw, "position")

	if !uriResult.Exists() || !posResult.Exists() {
		errObj = &ErrorObject{Code: InvalidParams, Message: "Missing textDocument.uri or position in completion params"}
		logger.Error("Invalid completion params", "error", errObj.Message)
		return // Response sent by defer
	}

	// Now unmarshal the full params as we need most of it
	if err := json.Unmarshal(paramsRaw, &params); err != nil {
		errObj = &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid params for completion: %v", err)}
		logger.Error("Failed to unmarshal completion params", "error", err)
		return // Response sent by defer
	}

	// Add URI to logger context
	logger = logger.With("uri", string(params.TextDocument.URI))

	// Log context if present
	if params.Context != nil {
		logger.Debug("Completion context info", "triggerKind", params.Context.TriggerKind, "triggerChar", params.Context.TriggerCharacter)
	}

	// --- Core Completion Logic ---
	docStoreMutex.RLock()
	docInfo, ok := documentStore[params.TextDocument.URI]
	var contentBytes []byte
	versionAtRequestStart := -1
	if ok {
		contentBytes = make([]byte, len(docInfo.Content))
		copy(contentBytes, docInfo.Content)
		versionAtRequestStart = docInfo.Version
	}
	docStoreMutex.RUnlock()

	if !ok {
		logger.Error("Document not found for completion")
		errObj = &ErrorObject{Code: RequestFailed, Message: fmt.Sprintf("Document not found: %s", params.TextDocument.URI)}
		return // Response sent by defer
	}

	// Check for cancellation before potentially long operation
	select {
	case <-reqCtx.Done():
		logger.Warn("Completion request cancelled before analysis.")
		errObj = &ErrorObject{Code: RequestCancelled, Message: "Request cancelled"}
		return // Response sent by defer
	default:
	}

	// Convert position
	lspPos := deepcomplete.LSPPosition{Line: params.Position.Line, Character: params.Position.Character}
	line, col, byteOffset, err := deepcomplete.LspPositionToBytePosition(contentBytes, lspPos)
	if err != nil {
		logger.Error("Error converting LSP position", "error", err)
		errObj = &ErrorObject{Code: RequestFailed, Message: fmt.Sprintf("Failed to convert position: %v", err)}
		return // Response sent by defer
	}

	// Cycle 2: Check document version *before* calling completer
	docStoreMutex.RLock()
	currentDocInfo, stillOk := documentStore[params.TextDocument.URI]
	docStoreMutex.RUnlock()

	if !stillOk {
		logger.Error("Document closed during completion request.")
		errObj = &ErrorObject{Code: RequestCancelled, Message: "Document closed during request"}
		return // Response sent by defer
	}

	if versionAtRequestStart != -1 && currentDocInfo.Version != versionAtRequestStart {
		logger.Warn("Document changed during completion request. Cancelling.", "startVersion", versionAtRequestStart, "currentVersion", currentDocInfo.Version)
		errObj = &ErrorObject{Code: RequestCancelled, Message: "Document changed during request"}
		return // Response sent by defer
	}
	// --- End Version Check ---

	// Check for cancellation again after version check, before LLM call
	select {
	case <-reqCtx.Done():
		logger.Warn("Completion request cancelled before LLM call.")
		errObj = &ErrorObject{Code: RequestCancelled, Message: "Request cancelled"}
		return // Response sent by defer
	default:
	}

	// Call core completer (potentially long running)
	logger.Debug("Calling GetCompletionStreamFromFile", "line", line, "col", col)
	var completionBuf bytes.Buffer
	startTime := time.Now()
	err = completer.GetCompletionStreamFromFile(reqCtx, string(params.TextDocument.URI), line, col, &completionBuf)
	duration := time.Since(startTime)
	logger.Debug("GetCompletionStreamFromFile finished", "duration", duration, "error", err)

	// Handle potential errors from the completer
	if err != nil {
		// Check if the error was due to cancellation
		if errors.Is(err, context.Canceled) || errors.Is(reqCtx.Err(), context.Canceled) {
			logger.Warn("Completion request cancelled during GetCompletionStreamFromFile.")
			errObj = &ErrorObject{Code: RequestCancelled, Message: "Request cancelled"}
		} else {
			logger.Error("Error getting completion from core", "line", line, "col", col, "error", err)
			errObj = &ErrorObject{Code: RequestFailed, Message: fmt.Sprintf("Completion failed: %v", err)}
		}
		return // Response sent by defer
	}

	// Format completion item if successful
	completionText := completionBuf.String()
	if completionText != "" {
		// --- Cycle 3: Filter based on subsequent text ---
		keepCompletion := true
		if byteOffset >= 0 && byteOffset <= len(contentBytes) {
			maxSubsequentLen := 50
			endSubsequent := byteOffset + maxSubsequentLen
			if endSubsequent > len(contentBytes) {
				endSubsequent = len(contentBytes)
			}
			subsequentBytes := contentBytes[byteOffset:endSubsequent]
			subsequentText := string(subsequentBytes)

			if len(subsequentText) > 0 {
				if !match.Match(completionText, subsequentText+"*") {
					logger.Info("Filtering completion due to subsequent text mismatch",
						"completion_start", firstN(completionText, 20),
						"subsequent_text", firstN(subsequentText, 20))
					keepCompletion = false
				} else {
					logger.Debug("Completion matches subsequent text",
						"completion_start", firstN(completionText, 20),
						"subsequent_text", firstN(subsequentText, 20))
				}
			}
		} else {
			logger.Warn("Invalid byte offset calculated for completion filtering", "offset", byteOffset)
		}
		// --- End Cycle 3 Filtering ---

		if keepCompletion {
			label := strings.TrimSpace(completionText)
			if firstNewline := strings.Index(label, "\n"); firstNewline != -1 {
				label = label[:firstNewline]
			}
			if len(label) > 50 {
				label = label[:50] + "..."
			}
			kind := CompletionItemKindText
			trimmedCompletion := strings.TrimSpace(completionText)
			if strings.HasPrefix(trimmedCompletion, "func ") {
				kind = CompletionItemKindFunction
			} else if strings.HasPrefix(trimmedCompletion, "var ") || strings.HasPrefix(trimmedCompletion, "const ") {
				kind = CompletionItemKindVariable
			} else if strings.HasPrefix(trimmedCompletion, "type ") {
				kind = CompletionItemKindKeyword
			}

			item := CompletionItem{
				Label:         label,
				Kind:          kind,
				Detail:        "DeepComplete Suggestion",
				Documentation: "Generated by DeepComplete",
			}
			if clientSupportsSnippets {
				item.InsertTextFormat = SnippetFormat
				item.InsertText = completionText + "$0"
			} else {
				item.InsertTextFormat = PlainTextFormat
				item.InsertText = completionText
			}
			result = CompletionList{IsIncomplete: false, Items: []CompletionItem{item}}
		} else {
			result = CompletionList{IsIncomplete: false, Items: []CompletionItem{}}
		}

	} else {
		result = CompletionList{IsIncomplete: false, Items: []CompletionItem{}} // Empty result
	}

	// Response is sent by the deferred function
}

// --- Notification Handlers (Run by Dispatcher or Quick Worker) ---

// Cycle 3: Added contextual logger argument
func handleInitializedNotification(ctx context.Context, event InitializedNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'initialized' notification.")
}

// Cycle 3: Added contextual logger argument
func handleExitNotification(ctx context.Context, event ExitNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'exit' notification. Signaling shutdown.")
	select {
	case <-shutdownChan: // Already shutting down
	default:
		close(shutdownChan)
	}
}

// Cycle 3: Added contextual logger argument
func handleDidOpenNotification(ctx context.Context, event DidOpenNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'textDocument/didOpen' notification...")
	var params DidOpenTextDocumentParams
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Error decoding didOpen params", "error", err)
		return
	}

	logger = logger.With("uri", string(params.TextDocument.URI)) // Add URI context

	docStoreMutex.Lock()
	documentStore[params.TextDocument.URI] = VersionedContent{
		Content: []byte(params.TextDocument.Text),
		Version: params.TextDocument.Version,
	}
	docStoreMutex.Unlock()

	logger.Info("Opened and stored document", "version", params.TextDocument.Version, "language", params.TextDocument.LanguageID, "length", len(params.TextDocument.Text))
}

// Cycle 3: Added contextual logger argument
func handleDidCloseNotification(ctx context.Context, event DidCloseNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'textDocument/didClose' notification...")
	var params DidCloseTextDocumentParams
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Error decoding didClose params", "error", err)
		return
	}

	logger = logger.With("uri", string(params.TextDocument.URI)) // Add URI context

	docStoreMutex.Lock()
	delete(documentStore, params.TextDocument.URI)
	docStoreMutex.Unlock()

	logger.Info("Closed and removed document")
}

// Cycle 3: Added contextual logger argument
func handleDidChangeNotification(ctx context.Context, event DidChangeNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'textDocument/didChange' notification...")
	var params DidChangeTextDocumentParams
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Error decoding didChange params", "error", err)
		return
	}

	logger = logger.With("uri", string(params.TextDocument.URI), "newVersion", params.TextDocument.Version) // Add URI/Version context

	if len(params.ContentChanges) != 1 {
		logger.Warn("Expected exactly one content change for full sync", "changes_count", len(params.ContentChanges))
		if len(params.ContentChanges) == 0 {
			return
		}
	}
	newText := params.ContentChanges[0].Text

	docStoreMutex.Lock()
	documentStore[params.TextDocument.URI] = VersionedContent{
		Content: []byte(newText),
		Version: params.TextDocument.Version,
	}
	docStoreMutex.Unlock()

	logger.Info("Updated document", "newLength", len(newText))

	// Cancel ongoing requests for this document as they are now stale
	cancelActiveRequestsForURI(params.TextDocument.URI, logger) // Pass logger

	// Invalidate cache
	if completer != nil {
		uriStr := string(params.TextDocument.URI)
		var dir string
		if strings.HasPrefix(uriStr, "file://") {
			parsedURL, err := url.Parse(uriStr)
			if err == nil {
				dir = filepath.Dir(parsedURL.Path)
			} else {
				logger.Warn("Could not parse file URI for cache invalidation", "uri", uriStr, "error", err)
				return
			}
		} else {
			dir = filepath.Dir(uriStr)
		}

		// Run invalidation in background
		go func(d string) {
			logger := slog.Default().With("uri", uriStr, "dir", d) // Create logger for goroutine
			logger.Debug("Attempting cache invalidation")
			if err := completer.InvalidateAnalyzerCache(d); err != nil {
				logger.Error("Error invalidating cache after didChange", "error", err)
			} else {
				logger.Debug("Cache invalidated due to didChange.")
			}
		}(dir)
	}
}

// Cycle 3: Added contextual logger argument
func handleDidChangeConfigurationNotification(ctx context.Context, event DidChangeConfigurationNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'workspace/didChangeConfiguration' notification...")
	settingsRaw := event.GetRawParams()

	logger.Debug("Received configuration change notification", "raw_settings", string(settingsRaw))

	if completer == nil {
		logger.Warn("Cannot apply configuration changes, completer not initialized.")
		return
	}

	newConfig := completer.GetCurrentConfig()
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
		}
	})

	if configUpdated {
		if err := completer.UpdateConfig(newConfig); err != nil {
			logger.Error("Error applying updated configuration", "error", err)
		} else {
			logger.Info("Successfully applied updated configuration from client.")
		}
	} else {
		logger.Info("No relevant configuration changes detected in notification.")
	}
}

// Cycle 3: Added contextual logger argument
func handleCancelRequestNotification(ctx context.Context, event CancelRequestEvent, logger *slog.Logger) {
	logger.Info("Handling '$/cancelRequest' notification...")
	idResult := gjson.GetBytes(event.GetRawParams(), "id")
	if !idResult.Exists() {
		logger.Warn("Received cancel request without ID.")
		return
	}
	reqID := idResult.Value()

	logger = logger.With("cancelledRequestID", reqID)
	logger.Info("Received cancellation request")

	activeRequestMut.Lock()
	cancelFunc, ok := activeRequests[reqID]
	delete(activeRequests, reqID) // Remove entry regardless
	activeRequestMut.Unlock()

	if ok {
		cancelFunc() // Call the cancellation function
		logger.Info("Cancelled active request")
	} else {
		logger.Warn("No active request found to cancel (already finished or invalid ID)")
	}
}

// ============================================================================
// Request Cancellation Helpers
// ============================================================================

// Cycle 3: Added contextual logger argument
func registerRequest(id any, cancel context.CancelFunc, logger *slog.Logger) {
	if id == nil {
		return // Cannot cancel requests without IDs
	}
	activeRequestMut.Lock()
	defer activeRequestMut.Unlock()

	if _, exists := activeRequests[id]; exists {
		logger.Warn("Request ID already registered for cancellation", "id", id)
	}

	activeRequests[id] = cancel
	logger.Debug("Registered request for cancellation")
}

// Cycle 3: Added contextual logger argument
func unregisterRequest(id any, logger *slog.Logger) {
	if id == nil {
		return
	}
	activeRequestMut.Lock()
	defer activeRequestMut.Unlock()
	if _, ok := activeRequests[id]; ok {
		delete(activeRequests, id)
		logger.Debug("Unregistered request")
	}
}

// Cycle 3: Added contextual logger argument
func cancelActiveRequestsForURI(uri DocumentURI, logger *slog.Logger) {
	logger.Warn("Cancellation by URI is not implemented efficiently yet.", "uri", uri)
	// TODO: Improve cancellation tracking to associate requests with URIs.
}

// ============================================================================
// I/O Helpers
// ============================================================================

// readMessage reads a single JSON-RPC message based on Content-Length header.
func readMessage(reader *bufio.Reader) ([]byte, error) {
	var contentLength int = -1
	for { // Read headers.
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return nil, io.EOF
			}
			return nil, fmt.Errorf("failed reading header line: %w", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			_, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength)
			if err != nil {
				// Use slog if available (might not be if error happens before setup)
				slog.Warn("Failed to parse Content-Length header", "header", line, "error", err)
			}
		}
	}
	if contentLength < 0 {
		return nil, fmt.Errorf("missing or invalid Content-Length header")
	}
	if contentLength == 0 {
		return []byte{}, nil
	}

	body := make([]byte, contentLength)
	n, err := io.ReadFull(reader, body)
	if err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("failed reading body (read %d/%d bytes): %w", n, contentLength, err)
	}
	return body, nil
}

// writeMessage encodes a message to JSON and writes it to the writer with headers.
// Assumes it's called by the single Response Writer goroutine.
func writeMessage(writer io.Writer, message interface{}) error {
	content, err := json.Marshal(message)
	if err != nil {
		slog.Error("Error marshalling message for output", "error", err, "messageType", fmt.Sprintf("%T", message))
		return fmt.Errorf("error marshalling message: %w", err)
	}
	slog.Debug("Sending message", "content", string(content))
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(content))

	if _, err := writer.Write([]byte(header)); err != nil {
		slog.Error("Error writing message header", "error", err)
		return fmt.Errorf("error writing header: %w", err)
	}
	if _, err := writer.Write(content); err != nil {
		slog.Error("Error writing message content", "error", err)
		return fmt.Errorf("error writing content: %w", err)
	}
	return nil
}

// Helper function to truncate strings for logging
func firstN(s string, n int) string {
	if len(s) > n {
		// Ensure n is valid index if string contains multi-byte runes
		if n < 0 {
			n = 0
		}
		i := 0
		for j := range s {
			if i == n {
				return s[:j] + "..."
			}
			i++
		}
		// If loop finishes, means n >= len(s)
		return s
	}
	return s
}
