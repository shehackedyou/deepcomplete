package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"expvar" // Cycle 3: Added expvar
	"fmt"
	"go/types"
	"io"
	"log"              // Used only for initial fatal errors
	"log/slog"         // Cycle 3: Added slog
	"net/http"         // Cycle 3: Added http
	_ "net/http/pprof" // Cycle 3: Added pprof endpoint registration
	"os"
	"path/filepath"
	"runtime"       // Cycle 3: Added runtime
	"runtime/debug" // For panic recovery stack traces
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson" // Cycle 2: Added gjson dependency
	"github.com/tidwall/match" // Cycle 5: Added match dependency

	// NOTE: Replace with your actual module path
	// Assumes deepcomplete package and its subpackages (like utils) exist
	"github.com/shehackedyou/deepcomplete"
	// "github.com/shehackedyou/deepcomplete/utils" // Conceptual: If path validation moved to utils
)

// ============================================================================
// Global Variables & Constants
// ============================================================================

// DocumentURI represents the URI for a text document. Defined here in main.
type DocumentURI string

// VersionedContent stores document content along with its version and access time.
type VersionedContent struct {
	Content    []byte
	Version    int
	LastAccess time.Time // Added for LRU eviction logic (conceptual)
}

var (
	// completer is initialized in main and used by handlers.
	completer *deepcomplete.DeepCompleter

	// documentStore holds the content and version of open documents, keyed by URI.
	documentStore = make(map[DocumentURI]VersionedContent)
	docStoreMutex sync.RWMutex // Protects concurrent access to documentStore.

	// clientSupportsSnippets tracks if the connected client supports snippet format. (Set during initialize).
	clientSupportsSnippets bool

	// --- Event-Driven Architecture Components ---
	eventQueue        chan Event
	responseQueue     chan ResponseWorkItem
	notificationQueue chan NotificationMessage // Added for window/showMessage
	shutdownChan      chan struct{}
	wg                sync.WaitGroup

	// activeRequests tracks ongoing requests that can be cancelled. Maps request ID to context cancel func.
	activeRequests   = make(map[any]context.CancelFunc)
	activeRequestMut sync.Mutex // Protects activeRequests map.

	// --- Configuration & Limits ---
	defaultQueueSize     = 100
	maxOpenDocuments     = 100              // Defensive: Limit number of docs in memory
	maxConcurrentWorkers = runtime.NumCPU() // Defensive: Limit concurrent handlers
	workerSemaphore      chan struct{}      // Defensive: Semaphore for limiting workers
	responseSendTimeout  = 2 * time.Second  // Defensive: Timeout for sending responses
	notifySendTimeout    = 1 * time.Second  // Defensive: Timeout for sending notifications

	// Cycle 3 & Defensive Programming: Monitoring variables
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
	workerSemaphoreTimeout   = expvar.NewInt("lsp.workerSemaphoreTimeout") // If timeout added
	responseQueueTimeout     = expvar.NewInt("lsp.responseQueueTimeout")
	notificationQueueTimeout = expvar.NewInt("lsp.notificationQueueTimeout")
	docsEvicted              = expvar.NewInt("lsp.docsEvicted")
)

// JSON-RPC Standard Error Codes
const (
	ParseError           int = -32700
	InvalidRequest       int = -32600
	MethodNotFound       int = -32601
	InvalidParams        int = -32602
	InternalError        int = -32603
	RequestCancelled     int = -32800
	ServerNotInitialized int = -32002 // Example: If request comes before 'initialize' finishes
	ServerBusy           int = -32000 // Example: Using generic code for busy/timeout
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

func (e BaseEvent) GetType() string               { return e.Type }
func (e BaseEvent) GetRequestID() any             { return e.RequestID }
func (e BaseEvent) GetRawParams() json.RawMessage { return e.RawParams }

// --- Request Events ---
type InitializeRequestEvent struct{ BaseEvent }
type ShutdownRequestEvent struct{ BaseEvent }
type CompletionRequestEvent struct{ BaseEvent }
type HoverRequestEvent struct{ BaseEvent } // Added Hover

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

// ============================================================================
// Middleware & Handler Types
// ============================================================================

// HandlerFunc defines the core logic for handling a specific LSP event.
// It returns the result (for requests) and any error encountered.
type HandlerFunc func(ctx context.Context, event Event, logger *slog.Logger) (result any, err *ErrorObject)

// MiddlewareFunc wraps a HandlerFunc to add pre/post processing logic.
type MiddlewareFunc func(next HandlerFunc) HandlerFunc

// ============================================================================
// Work Items & JSON-RPC Structures
// ============================================================================

// ResponseWorkItem holds the result or error for a completed request.
type ResponseWorkItem struct {
	RequestID any
	Result    any
	Error     *ErrorObject
}

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

// NotificationMessage used for parsing incoming and sending outgoing notifications.
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
// LSP Specific Structures
// ============================================================================

// Position represents a 0-based line/character offset.
type Position struct {
	Line      uint32 `json:"line"`      // 0-based
	Character uint32 `json:"character"` // 0-based, UTF-16 offset
}

// Range represents a range in a text document.
type Range struct {
	Start Position `json:"start"`
	End   Position `json:"end"`
}

// Location represents a location inside a resource, such as a line inside a text file.
type Location struct {
	URI   DocumentURI `json:"uri"`
	Range Range       `json:"range"`
}

// TextDocumentIdentifier identifies a specific text document.
type TextDocumentIdentifier struct {
	URI DocumentURI `json:"uri"`
}

// TextDocumentItem represents a text document.
type TextDocumentItem struct {
	URI        DocumentURI `json:"uri"`
	LanguageID string      `json:"languageId"`
	Version    int         `json:"version"` // Must be non-negative
	Text       string      `json:"text"`
}

// InitializeParams parameters for the initialize request.
type InitializeParams struct {
	ProcessID             int                `json:"processId,omitempty"`
	RootURI               DocumentURI        `json:"rootUri,omitempty"`
	ClientInfo            *ClientInfo        `json:"clientInfo,omitempty"`
	Capabilities          ClientCapabilities `json:"capabilities"`
	InitializationOptions json.RawMessage    `json:"initializationOptions,omitempty"`
}

// ClientInfo information about the client.
type ClientInfo struct {
	Name    string `json:"name,omitempty"`
	Version string `json:"version,omitempty"`
}

// ClientCapabilities capabilities provided by the client.
type ClientCapabilities struct {
	Workspace    *WorkspaceClientCapabilities    `json:"workspace,omitempty"`
	TextDocument *TextDocumentClientCapabilities `json:"textDocument,omitempty"`
}

// WorkspaceClientCapabilities workspace specific client capabilities.
type WorkspaceClientCapabilities struct {
	Configuration bool `json:"configuration,omitempty"`
	// Add other workspace capabilities if needed
}

// TextDocumentClientCapabilities text document specific client capabilities.
type TextDocumentClientCapabilities struct {
	Completion *CompletionClientCapabilities `json:"completion,omitempty"`
	Hover      *HoverClientCapabilities      `json:"hover,omitempty"` // Added Hover
	// Add other text document capabilities if needed
}

// CompletionClientCapabilities client capabilities for completion.
type CompletionClientCapabilities struct {
	CompletionItem *CompletionItemClientCapabilities `json:"completionItem,omitempty"`
}

// CompletionItemClientCapabilities client capabilities specific to completion items.
type CompletionItemClientCapabilities struct {
	SnippetSupport bool `json:"snippetSupport,omitempty"`
}

// HoverClientCapabilities client capabilities for hover.
type HoverClientCapabilities struct { // Added Hover
	ContentFormat []MarkupKind `json:"contentFormat,omitempty"` // e.g., ["markdown", "plaintext"]
}

// InitializeResult result of the initialize request.
type InitializeResult struct {
	Capabilities ServerCapabilities `json:"capabilities"`
	ServerInfo   *ServerInfo        `json:"serverInfo,omitempty"`
}

// ServerCapabilities capabilities provided by the server.
type ServerCapabilities struct {
	TextDocumentSync   *TextDocumentSyncOptions `json:"textDocumentSync,omitempty"`
	CompletionProvider *CompletionOptions       `json:"completionProvider,omitempty"` // Use options struct if needed
	HoverProvider      bool                     `json:"hoverProvider,omitempty"`      // Added Hover (can be HoverOptions{} too)
}

// TextDocumentSyncOptions options for text document synchronization.
type TextDocumentSyncOptions struct {
	OpenClose bool                 `json:"openClose,omitempty"`
	Change    TextDocumentSyncKind `json:"change,omitempty"` // Specifies how changes are synced (1=Full)
}

// TextDocumentSyncKind defines how text document changes are synced.
type TextDocumentSyncKind int

const (
	TextDocumentSyncKindNone TextDocumentSyncKind = 0
	TextDocumentSyncKindFull TextDocumentSyncKind = 1 // We only support Full sync
)

// CompletionOptions server completion capabilities.
type CompletionOptions struct {
	// Add triggerCharacters, resolveProvider etc. if needed
}

// ServerInfo information about the server.
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"` // Populated in main
}

// DidOpenTextDocumentParams parameters for textDocument/didOpen.
type DidOpenTextDocumentParams struct {
	TextDocument TextDocumentItem `json:"textDocument"`
}

// DidCloseTextDocumentParams parameters for textDocument/didClose.
type DidCloseTextDocumentParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

// DidChangeTextDocumentParams parameters for textDocument/didChange.
type DidChangeTextDocumentParams struct {
	TextDocument   VersionedTextDocumentIdentifier  `json:"textDocument"`
	ContentChanges []TextDocumentContentChangeEvent `json:"contentChanges"` // Array, but we only handle the last one for Full sync
}

// VersionedTextDocumentIdentifier identifies a text document with a version number.
type VersionedTextDocumentIdentifier struct {
	TextDocumentIdentifier
	Version int `json:"version"` // Must be non-negative
}

// TextDocumentContentChangeEvent an event describing a change to a text document.
type TextDocumentContentChangeEvent struct {
	// Range is omitted - we only support Full sync
	Text string `json:"text"` // The new full content of the document
}

// DidChangeConfigurationParams parameters for workspace/didChangeConfiguration.
type DidChangeConfigurationParams struct {
	Settings json.RawMessage `json:"settings"` // Can be anything, use gjson to parse needed parts
}

// CompletionParams parameters for textDocument/completion.
type CompletionParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     Position               `json:"position"`
	Context      *CompletionContext     `json:"context,omitempty"`
}

// CompletionContext additional information about the context in which completion request is triggered.
type CompletionContext struct {
	TriggerKind      CompletionTriggerKind `json:"triggerKind"`
	TriggerCharacter string                `json:"triggerCharacter,omitempty"`
}

// CompletionTriggerKind how completion was triggered.
type CompletionTriggerKind int

const (
	CompletionTriggerKindInvoked              CompletionTriggerKind = 1 // Invoked by user explicitly
	CompletionTriggerKindTriggerChar          CompletionTriggerKind = 2 // Triggered by typing a trigger character
	CompletionTriggerKindTriggerForIncomplete CompletionTriggerKind = 3 // Triggered again for incomplete list
)

// CompletionList represents a list of completion items.
type CompletionList struct {
	IsIncomplete bool             `json:"isIncomplete"` // We currently always return complete lists
	Items        []CompletionItem `json:"items"`
}

// CompletionItem represents a single completion suggestion.
type CompletionItem struct {
	Label            string             `json:"label"`                      // Text shown in list
	Kind             CompletionItemKind `json:"kind,omitempty"`             // Type of completion (function, variable, etc.)
	Detail           string             `json:"detail,omitempty"`           // Additional info (e.g., type signature)
	Documentation    string             `json:"documentation,omitempty"`    // Documentation string (can be MarkupContent later)
	InsertTextFormat InsertTextFormat   `json:"insertTextFormat,omitempty"` // PlainText or Snippet
	InsertText       string             `json:"insertText,omitempty"`       // Text to insert
}

// CompletionItemKind defines the kind of completion item.
type CompletionItemKind int // Standard LSP kinds

const (
	CompletionItemKindText     CompletionItemKind = 1
	CompletionItemKindMethod   CompletionItemKind = 2
	CompletionItemKindFunction CompletionItemKind = 3
	CompletionItemKindVariable CompletionItemKind = 6
	CompletionItemKindField    CompletionItemKind = 5 // Added Field
	CompletionItemKindStruct   CompletionItemKind = 7 // Added Struct
	CompletionItemKindKeyword  CompletionItemKind = 14
	CompletionItemKindModule   CompletionItemKind = 9 // Added Module/Package
)

// InsertTextFormat defines the format of the insert text.
type InsertTextFormat int

const (
	PlainTextFormat InsertTextFormat = 1
	SnippetFormat   InsertTextFormat = 2
)

// CancelParams parameters for $/cancelRequest.
type CancelParams struct {
	ID any `json:"id"` // ID of the request to cancel (number or string)
}

// Added Hover Structures
type HoverParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     Position               `json:"position"`
}
type HoverResult struct {
	Contents MarkupContent `json:"contents"`
	Range    *Range        `json:"range,omitempty"` // Optional: range of the hovered symbol
}
type MarkupContent struct {
	Kind  MarkupKind `json:"kind"` // e.g., "markdown" or "plaintext"
	Value string     `json:"value"`
}
type MarkupKind string

const (
	MarkupKindPlainText MarkupKind = "plaintext"
	MarkupKindMarkdown  MarkupKind = "markdown"
)

// Added window/showMessage Structures
type MessageType int

const (
	MessageTypeError   MessageType = 1
	MessageTypeWarning MessageType = 2
	MessageTypeInfo    MessageType = 3
	MessageTypeLog     MessageType = 4
)

type ShowMessageParams struct {
	Type    MessageType `json:"type"`
	Message string      `json:"message"`
}

// ============================================================================
// Main Server Logic & Event Loop
// ============================================================================

// App version (set via linker flags -ldflags="-X main.appVersion=...")
var appVersion = "dev"

func main() {
	// Setup logging *before* initializing slog
	logFile, err := os.OpenFile("deepcomplete-lsp.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0660)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err) // Use standard log for initial fatal error
	}
	defer logFile.Close()

	// Initialize slog (Cycle 3)
	logLevel := slog.LevelDebug // TODO: Make configurable via flag/env
	logWriter := io.MultiWriter(os.Stderr, logFile)
	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: true}
	handler := slog.NewTextHandler(logWriter, &handlerOpts)
	logger := slog.New(handler)
	slog.SetDefault(logger)

	slog.Info("DeepComplete LSP server starting...", "version", appVersion)

	// Enable profiling rates (Cycle 3)
	runtime.SetBlockProfileRate(1)
	runtime.SetMutexProfileFraction(1)
	slog.Info("Enabled block and mutex profiling")

	// Start pprof/expvar HTTP server (Cycle 3)
	debugListenAddr := "localhost:6061" // TODO: Make configurable?
	go func() {
		slog.Info("Starting debug server for pprof/expvar", "addr", debugListenAddr)
		// Use a separate ServeMux for debug endpoints
		debugMux := http.NewServeMux()
		// Register pprof handlers under /debug/pprof/
		debugMux.HandleFunc("/debug/pprof/", http.DefaultServeMux.ServeHTTP) // Delegate to default mux which has pprof handlers
		debugMux.HandleFunc("/debug/pprof/cmdline", http.DefaultServeMux.ServeHTTP)
		debugMux.HandleFunc("/debug/pprof/profile", http.DefaultServeMux.ServeHTTP)
		debugMux.HandleFunc("/debug/pprof/symbol", http.DefaultServeMux.ServeHTTP)
		debugMux.HandleFunc("/debug/pprof/trace", http.DefaultServeMux.ServeHTTP)
		// Register expvar handler at /debug/vars
		debugMux.HandleFunc("/debug/vars", expvar.Handler().ServeHTTP)
		if err := http.ListenAndServe(debugListenAddr, debugMux); err != nil {
			slog.Error("Debug server failed", "error", err)
		}
	}()

	// Initialize core service
	var initErr error
	completer, initErr = deepcomplete.NewDeepCompleter()
	if initErr != nil {
		slog.Error("Failed to initialize DeepCompleter service", "error", initErr)
		if errors.Is(initErr, deepcomplete.ErrConfig) {
			// Try to send notification even if completer init failed due to config warnings
			sendShowMessageNotification(MessageTypeWarning, fmt.Sprintf("Configuration loaded with warnings: %v", initErr))
		} else {
			// Treat other init errors as fatal for now
			os.Exit(1)
		}
		// Continue if it was just a config warning, completer might use defaults
		if completer == nil {
			os.Exit(1) // Exit if completer is still nil
		}
	}
	defer func() {
		slog.Info("Closing DeepCompleter service...")
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()
	slog.Info("DeepComplete service initialized.")

	// Initialize queues, shutdown channel, and semaphore
	eventQueue = make(chan Event, defaultQueueSize)
	responseQueue = make(chan ResponseWorkItem, defaultQueueSize)
	notificationQueue = make(chan NotificationMessage, 20) // Smaller buffer for notifications
	shutdownChan = make(chan struct{})
	workerSemaphore = make(chan struct{}, maxConcurrentWorkers) // Initialize semaphore
	slog.Info("Initialized queues and semaphore", "queueSize", defaultQueueSize, "maxWorkers", maxConcurrentWorkers)

	// Publish expvar metrics (Cycle 3 & 9 & Defensive)
	publishExpvarMetrics()

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

// publishExpvarMetrics sets up expvar metrics publishing.
func publishExpvarMetrics() {
	expvar.Publish("eventQueueLength", expvar.Func(func() any {
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
	expvar.Publish("notificationQueueLength", expvar.Func(func() any {
		if notificationQueue == nil {
			return 0
		}
		return len(notificationQueue)
	}))
	expvar.Publish("activeRequests", expvar.Func(func() any {
		activeRequestMut.Lock()
		defer activeRequestMut.Unlock()
		return len(activeRequests)
	}))
	expvar.Publish("activeWorkers", expvar.Func(func() any {
		// Len of semaphore channel indicates number of *available* slots
		// So, capacity - len = active workers
		if workerSemaphore == nil {
			return 0
		}
		return maxConcurrentWorkers - len(workerSemaphore)
	}))
	expvar.Publish("openDocuments", expvar.Func(func() any {
		docStoreMutex.RLock()
		defer docStoreMutex.RUnlock()
		return len(documentStore)
	}))

	// Publish Ristretto metrics if cache is enabled (Cycle 9)
	// Assumes completer is initialized before this is called, or check completer != nil
	if completer != nil {
		if analyzer := completer.GetAnalyzer(); analyzer != nil && analyzer.MemoryCacheEnabled() {
			expvar.Publish("ristrettoHits", expvar.Func(func() any { return analyzer.GetMemoryCacheMetrics().Hits() }))
			expvar.Publish("ristrettoMisses", expvar.Func(func() any { return analyzer.GetMemoryCacheMetrics().Misses() }))
			expvar.Publish("ristrettoKeysAdded", expvar.Func(func() any { return analyzer.GetMemoryCacheMetrics().KeysAdded() }))
			expvar.Publish("ristrettoCostAdded", expvar.Func(func() any { return analyzer.GetMemoryCacheMetrics().CostAdded() }))
			expvar.Publish("ristrettoCostEvicted", expvar.Func(func() any { return analyzer.GetMemoryCacheMetrics().CostEvicted() }))
			expvar.Publish("ristrettoRatio", expvar.Func(func() any { return analyzer.GetMemoryCacheMetrics().Ratio() }))
		}
	}
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
			return
		default:
			content, err := readMessage(reader) // Uses slog internally for warnings
			if err != nil {
				if errors.Is(err, io.EOF) {
					slog.Info("Client closed connection (EOF). Stopping reader and signaling shutdown.")
					select {
					case <-shutdownChan: // Avoid closing already closed channel
					default:
						close(shutdownChan)
					}
					return
				}
				// Log other read errors and potentially continue/shutdown
				slog.Error("Error reading message", "error", err)
				// Depending on the error, might signal shutdown or just continue
				// For now, continue after a short pause
				time.Sleep(100 * time.Millisecond)
				continue
			}

			eventsReceived.Add(1)
			slog.Debug("Received raw message", "bytes", len(content)) // Avoid logging full content at debug level
			event := parseMessageToEvent(content)                     // Uses slog internally
			if event == nil {
				eventsIgnored.Add(1) // Ignored unknown notification
				continue
			}
			// Log parsed event type and ID (if any)
			logAttrs := []any{slog.String("type", event.GetType())}
			if reqID := event.GetRequestID(); reqID != nil {
				logAttrs = append(logAttrs, slog.Any("requestID", reqID))
			}
			slog.Debug("Parsed event", logAttrs...)

			// Send event to queue with timeout
			select {
			case eventQueue <- event:
				// Event sent successfully
			case <-shutdownChan:
				slog.Warn("Shutdown signaled while sending event to queue.")
				return
			case <-time.After(5 * time.Second): // Timeout for sending to queue
				slog.Error("Timeout sending event to eventQueue (dispatcher slow/stuck?)", "eventType", event.GetType())
				// Consider dropping event or signaling critical error/shutdown
			}
		}
	}
}

// parseMessageToEvent parses raw message content into a specific Event struct.
func parseMessageToEvent(content []byte) Event {
	// Use gjson for efficient initial parsing
	methodResult := gjson.GetBytes(content, "method")
	if !methodResult.Exists() {
		slog.Error("Received message without method", "content_preview", firstN(string(content), 100))
		// Cannot determine type, return minimal UnknownEvent
		return UnknownEvent{BaseEvent: BaseEvent{RawParams: content}, ParseError: errors.New("message missing method field")}
	}
	method := methodResult.String()
	idResult := gjson.GetBytes(content, "id")
	// Get raw params bytes *once*
	rawParams := json.RawMessage(gjson.GetBytes(content, "params").Raw)

	// Create BaseEvent
	base := BaseEvent{Type: method, RawParams: rawParams}

	if idResult.Exists() { // Request
		base.RequestID = idResult.Value() // Store the ID (number or string)
		switch method {
		case "initialize":
			return InitializeRequestEvent{BaseEvent: base}
		case "shutdown":
			return ShutdownRequestEvent{BaseEvent: base}
		case "textDocument/completion":
			return CompletionRequestEvent{BaseEvent: base}
		case "textDocument/hover": // Added Hover
			return HoverRequestEvent{BaseEvent: base}
		default:
			slog.Warn("Received unknown request method", "method", method, "requestID", base.RequestID)
			// Return UnknownEvent for unhandled requests
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
			slog.Debug("Ignoring unknown notification method", "method", method)
			// Return nil to indicate dispatcher should ignore it
			return nil
		}
	}
}

// runDispatcher reads events, applies middleware, and spawns workers.
func runDispatcher() {
	defer wg.Done()
	defer func() {
		if r := recover(); r != nil {
			panicsRecovered.Add(1)
			slog.Error("PANIC in runDispatcher", "error", r, "stack", string(debug.Stack()))
		}
		slog.Info("Dispatcher goroutine stopped.")
	}()

	serverCtx, cancelServer := context.WithCancel(context.Background())
	defer cancelServer()

	// Define Middleware Chain
	middlewares := []MiddlewareFunc{
		panicRecoveryMiddleware, // Innermost: Catch panics from handlers
		loggingMiddleware,       // Log entry/exit/duration
		cancellationMiddleware,  // Handle cancellation context for requests
		// Add more middleware here (e.g., validation, rate limiting)
	}

	for {
		select {
		case <-shutdownChan:
			slog.Info("Dispatcher received shutdown signal.")
			cancelServer()
			// Allow some time for existing workers to finish?
			// time.Sleep(200 * time.Millisecond)
			return
		case event, ok := <-eventQueue:
			if !ok {
				slog.Info("Event queue closed. Stopping dispatcher.")
				cancelServer()
				return
			}

			eventLogger := slog.Default().With(
				"requestID", event.GetRequestID(),
				"eventType", fmt.Sprintf("%T", event),
			)
			eventLogger.Debug("Dispatcher received event") // Changed level to Debug

			// --- Select Base Handler and Apply Middleware ---
			var baseHandler HandlerFunc
			isResourceIntensive := false // Flag for semaphore limiting
			isNotification := event.GetRequestID() == nil

			switch event.(type) {
			// --- Requests ---
			case InitializeRequestEvent:
				baseHandler = initializeHandler
				isResourceIntensive = false // Initialize is usually quick
			case ShutdownRequestEvent:
				baseHandler = shutdownHandler
				isResourceIntensive = false
			case CompletionRequestEvent:
				baseHandler = completionHandler
				isResourceIntensive = true // Calls Analyze + LLM
			case HoverRequestEvent:
				baseHandler = hoverHandler
				isResourceIntensive = true // Calls Analyze
			// --- Notifications ---
			case InitializedNotificationEvent:
				go handleInitializedNotification(serverCtx, event.(InitializedNotificationEvent), eventLogger.With("handler", "Initialized"))
				notificationsHandled.Add(1)
				continue // Handle simple notifications directly
			case ExitNotificationEvent:
				go handleExitNotification(serverCtx, event.(ExitNotificationEvent), eventLogger.With("handler", "Exit"))
				notificationsHandled.Add(1)
				continue
			case DidOpenNotificationEvent:
				go handleDidOpenNotification(serverCtx, event.(DidOpenNotificationEvent), eventLogger.With("handler", "DidOpen"))
				notificationsHandled.Add(1)
				continue
			case DidCloseNotificationEvent:
				go handleDidCloseNotification(serverCtx, event.(DidCloseNotificationEvent), eventLogger.With("handler", "DidClose"))
				notificationsHandled.Add(1)
				continue
			case DidChangeNotificationEvent:
				go handleDidChangeNotification(serverCtx, event.(DidChangeNotificationEvent), eventLogger.With("handler", "DidChange"))
				notificationsHandled.Add(1)
				continue
			case DidChangeConfigurationNotificationEvent:
				go handleDidChangeConfigurationNotification(serverCtx, event.(DidChangeConfigurationNotificationEvent), eventLogger.With("handler", "DidChangeConfiguration"))
				notificationsHandled.Add(1)
				continue
			case CancelRequestEvent:
				go handleCancelRequestNotification(serverCtx, event.(CancelRequestEvent), eventLogger.With("handler", "CancelRequest"))
				notificationsHandled.Add(1)
				continue
			// --- Error/Unknown ---
			case UnknownEvent:
				eventLogger.Warn("Dispatcher received UnknownEvent", "error", event.(UnknownEvent).ParseError)
				if reqID := event.GetRequestID(); reqID != nil {
					errorsReported.Add(1)
					// Send error response directly for unknown request methods
					sendResponse(reqID, nil, &ErrorObject{Code: MethodNotFound, Message: fmt.Sprintf("Method not found: %s", event.GetType())}, serverCtx, eventLogger)
				}
				continue // Skip middleware for unknown events
			default:
				eventLogger.Warn("Dispatcher received unhandled event type")
				continue // Skip middleware for unhandled events
			}

			// Apply middleware chain only to requests with defined handlers
			if baseHandler != nil {
				requestsDispatched.Add(1) // Count dispatched requests
				wrappedHandler := baseHandler
				for i := len(middlewares) - 1; i >= 0; i-- {
					// Skip cancellation middleware if it's a notification (though we handle notifications separately now)
					// Use runtime reflection to check function pointer equality robustly if needed,
					// but comparing function variables directly often works for defined functions.
					isCancellationMw := fmt.Sprintf("%p", middlewares[i]) == fmt.Sprintf("%p", cancellationMiddleware)
					if isNotification && isCancellationMw {
						continue
					}
					wrappedHandler = middlewares[i](wrappedHandler)
				}

				// Spawn worker goroutine with semaphore limiting
				go func(handler HandlerFunc, reqEvent Event, reqLogger *slog.Logger, limit bool) {
					if limit {
						workerSemaphoreAcqTry.Add(1)
						reqLogger.Debug("Acquiring worker semaphore slot...")
						select {
						case workerSemaphore <- struct{}{}:
							workerSemaphoreAcqOk.Add(1)
							reqLogger.Debug("Worker semaphore slot acquired.")
							defer func() {
								<-workerSemaphore
								reqLogger.Debug("Worker semaphore slot released.")
							}()
						case <-serverCtx.Done():
							reqLogger.Warn("Server shutdown while waiting for worker semaphore.")
							// Send cancellation response if possible
							if reqID := reqEvent.GetRequestID(); reqID != nil {
								sendResponse(reqID, nil, &ErrorObject{Code: RequestCancelled, Message: "Server shutting down"}, serverCtx, reqLogger)
							}
							return // Don't proceed if server is shutting down
							// Optional: Timeout for acquiring semaphore
							// case <-time.After(10 * time.Second):
							//    workerSemaphoreTimeout.Add(1)
							//    reqLogger.Error("Timeout waiting for worker semaphore slot.")
							//    sendResponse(reqEvent.GetRequestID(), nil, &ErrorObject{Code: ServerBusy, Message: "Server busy, please try again later."}, serverCtx, reqLogger)
							//    return
						}
					} else {
						reqLogger.Debug("Skipping semaphore for non-intensive handler.")
					}

					// Execute the fully wrapped handler
					// Pass serverCtx here, cancellationMiddleware creates reqCtx internally
					res, errObj := handler(serverCtx, reqEvent, reqLogger)

					// Send response only if it was a request
					if reqID := reqEvent.GetRequestID(); reqID != nil {
						sendResponse(reqID, res, errObj, serverCtx, reqLogger)
					} else if errObj != nil {
						// Log errors from notification handlers if they somehow return one
						reqLogger.Error("Error occurred processing notification within middleware", "error", errObj)
						errorsReported.Add(1)
					}
				}(wrappedHandler, ev, eventLogger, isResourceIntensive) // Pass copies
			}
		}
	}
}

// runResponseWriter reads completed work items and notifications, writing them to the client.
func runResponseWriter(w io.Writer) {
	defer wg.Done()
	defer func() {
		if r := recover(); r != nil {
			panicsRecovered.Add(1)
			slog.Error("PANIC in runResponseWriter", "error", r, "stack", string(debug.Stack()))
		}
		slog.Info("Response Writer goroutine stopped.")
	}()

	var writerMu sync.Mutex // Protect writes to w

	for {
		select {
		case <-shutdownChan:
			slog.Info("Response Writer received shutdown signal.")
			// Process remaining items? Optional, could lead to delays.
			// For now, just exit cleanly.
			return
		case workItem, ok := <-responseQueue:
			if !ok {
				slog.Info("Response queue closed.")
				// If response queue closes, should probably stop notification queue too?
				// Or let it drain if notifications are still being sent during shutdown prep.
				if notificationQueue == nil {
					return
				} // Exit if notification queue also closed/nil
				continue // Otherwise continue processing notifications
			}
			// Construct ResponseMessage
			response := ResponseMessage{
				JSONRPC: "2.0",
				ID:      workItem.RequestID,
				Result:  workItem.Result,
				Error:   workItem.Error,
			}
			// Write message
			writerMu.Lock()
			err := writeMessage(w, response)
			writerMu.Unlock()
			if err != nil {
				slog.Error("Error writing response", "requestID", workItem.RequestID, "error", err)
			} else {
				responsesSent.Add(1)
				slog.Debug("Successfully wrote response", "requestID", workItem.RequestID)
			}

		case notificationMsg, ok := <-notificationQueue: // Handle notifications
			if !ok {
				slog.Info("Notification queue closed.")
				notificationQueue = nil // Mark as closed
				if responseQueue == nil {
					return
				} // Exit if response queue also closed/nil
				continue // Otherwise continue processing responses
			}
			// Write notification message
			writerMu.Lock()
			err := writeMessage(w, notificationMsg) // writeMessage handles various types
			writerMu.Unlock()
			if err != nil {
				slog.Error("Error writing notification", "method", notificationMsg.Method, "error", err)
			} else {
				notificationsSent.Add(1)
				slog.Debug("Successfully wrote notification", "method", notificationMsg.Method)
			}
		}
	}
}

// sendResponse is a helper to send a ResponseWorkItem to the queue with timeout.
// Used by the middleware wrapper goroutine.
func sendResponse(reqID any, result any, errObj *ErrorObject, serverCtx context.Context, logger *slog.Logger) {
	if reqID == nil {
		logger.Warn("Attempted to send response for notification")
		return
	}
	workItem := ResponseWorkItem{RequestID: reqID, Result: result, Error: errObj}
	select {
	case responseQueue <- workItem:
		if errObj != nil {
			errorsReported.Add(1)
		}
		logger.Debug("Successfully sent response item to queue")
	case <-time.After(responseSendTimeout):
		responseQueueTimeout.Add(1)
		logger.Error("Timeout sending response to queue (writer blocked?) - dropping response.", "requestID", workItem.RequestID, "timeout", responseSendTimeout)
		errorsReported.Add(1)
	case <-serverCtx.Done(): // Check server context for shutdown
		logger.Warn("Shutdown occurred while attempting to send response.", "requestID", workItem.RequestID)
	}
}

// ============================================================================
// Middleware Implementations
// ============================================================================

func panicRecoveryMiddleware(next HandlerFunc) HandlerFunc {
	return func(ctx context.Context, event Event, logger *slog.Logger) (result any, err *ErrorObject) {
		defer func() {
			if r := recover(); r != nil {
				panicsRecovered.Add(1)
				stack := string(debug.Stack())
				logger.Error("PANIC recovered in handler", "error", r, "stack", stack)
				err = &ErrorObject{Code: InternalError, Message: fmt.Sprintf("Internal server error from panic: %v", r)}
				result = nil // Ensure result is nil on panic
			}
		}()
		result, err = next(ctx, event, logger) // Call the next handler
		return
	}
}

func loggingMiddleware(next HandlerFunc) HandlerFunc {
	return func(ctx context.Context, event Event, logger *slog.Logger) (result any, err *ErrorObject) {
		startTime := time.Now()
		// Use Debug for start, Info/Error for finish
		logger.Debug("Handler starting")

		// Call next handler
		result, err = next(ctx, event, logger)

		duration := time.Since(startTime)
		logLevel := slog.LevelInfo
		if err != nil {
			logLevel = slog.LevelError
		} // Log as error if handler returned error

		// Log completion with duration and error status
		logger.Log(ctx, logLevel, "Handler finished", slog.Duration("duration", duration), slog.Any("error", err))
		return
	}
}

func cancellationMiddleware(next HandlerFunc) HandlerFunc {
	return func(ctx context.Context, event Event, logger *slog.Logger) (result any, err *ErrorObject) {
		reqID := event.GetRequestID()
		// Should not be called for notifications due to dispatcher logic, but check defensively
		if reqID == nil {
			logger.Warn("Cancellation middleware called for notification event type", "type", event.GetType())
			return next(ctx, event, logger) // Pass through
		}

		// Create cancellable context for this specific request
		// Use the passed context (which should be serverCtx) as the parent
		reqCtx, cancel := context.WithCancel(ctx)
		registerRequest(reqID, cancel, logger)
		defer unregisterRequest(reqID, logger) // Ensure unregistration

		// Call next handler with the request-specific context
		result, err = next(reqCtx, event, logger)

		// Check if cancellation happened *during* the handler execution
		// Only override error if handler didn't already return one (e.g. from LLM call cancellation)
		if reqCtx.Err() != nil && err == nil {
			logger.Warn("Request cancelled during handler execution.", "error", reqCtx.Err())
			err = &ErrorObject{Code: RequestCancelled, Message: "Request cancelled"}
			result = nil // Ensure result is nil on cancellation
		}
		return
	}
}

// ============================================================================
// Core Handler Logic (Refactored for Middleware)
// ============================================================================

// initializeHandler handles the 'initialize' request.
func initializeHandler(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: initialize")
	var params InitializeParams
	if event.GetRawParams() == nil {
		return nil, &ErrorObject{Code: InvalidParams, Message: "Missing parameters for initialize"}
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Invalid params for initialize", "error", err)
		return nil, &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid params for initialize: %v", err)}
	}

	// Process params...
	if params.ClientInfo != nil {
		logger.Info("Client Info", "name", params.ClientInfo.Name, "version", params.ClientInfo.Version)
	}
	clientSupportsSnippets = false
	if params.Capabilities.TextDocument != nil &&
		params.Capabilities.TextDocument.Completion != nil &&
		params.Capabilities.TextDocument.Completion.CompletionItem != nil &&
		params.Capabilities.TextDocument.Completion.CompletionItem.SnippetSupport {
		clientSupportsSnippets = true
		logger.Info("Client supports snippets.")
	} else {
		logger.Info("Client does not support snippets.")
	}

	// Return capabilities
	initResult := InitializeResult{
		Capabilities: ServerCapabilities{
			TextDocumentSync:   &TextDocumentSyncOptions{OpenClose: true, Change: TextDocumentSyncKindFull},
			CompletionProvider: &CompletionOptions{},
			HoverProvider:      true, // Advertise hover support
		},
		ServerInfo: &ServerInfo{Name: "DeepComplete LSP", Version: appVersion},
	}
	return initResult, nil
}

// shutdownHandler handles the 'shutdown' request.
func shutdownHandler(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: shutdown")
	// Indicate successful preparation for shutdown. The actual shutdown happens on 'exit'.
	return nil, nil
}

// completionHandler handles 'textDocument/completion'.
func completionHandler(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: completion")
	var params CompletionParams

	// 1. Parse and Validate Parameters
	if event.GetRawParams() == nil {
		return nil, &ErrorObject{Code: InvalidParams, Message: "Missing parameters for completion"}
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Failed to unmarshal completion params", "error", err)
		return nil, &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid params for completion: %v", err)}
	}
	// Validate position
	if params.Position.Line < 0 || params.Position.Character < 0 {
		logger.Error("Invalid completion params: negative position", "line", params.Position.Line, "char", params.Position.Character)
		return nil, &ErrorObject{Code: InvalidParams, Message: "Invalid position: line and character must be non-negative."}
	}
	// Validate URI and get path
	absPath, pathErr := deepcomplete.ValidateAndGetFilePath(string(params.TextDocument.URI), logger) // Use string cast
	if pathErr != nil {
		logger.Error("Invalid completion params: bad URI", "uri", params.TextDocument.URI, "error", pathErr)
		return nil, &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid document URI: %v", pathErr)}
	}
	logger = logger.With("uri", string(params.TextDocument.URI), "path", absPath) // Add validated path to logger

	// 2. Get Document State
	docStoreMutex.RLock()
	docInfo, ok := documentStore[params.TextDocument.URI]
	// Update last access time if found
	if ok {
		docInfo.LastAccess = time.Now()
	} // Conceptual LRU update
	docStoreMutex.RUnlock() // Release read lock before potential write lock in LRU update

	// If LRU update needed write lock:
	// docStoreMutex.Lock()
	// if docInfoPtr, ok := documentStore[params.TextDocument.URI]; ok {
	//     docInfoPtr.LastAccess = time.Now()
	//     documentStore[params.TextDocument.URI] = docInfoPtr // Update map if value type
	// }
	// docStoreMutex.Unlock()

	if !ok {
		logger.Error("Document not found for completion")
		return nil, nil // Return nil result, not an error, for missing doc
	}

	// 3. Convert Position
	lspPos := deepcomplete.LSPPosition{Line: params.Position.Line, Character: params.Position.Character}
	line, col, byteOffset, posErr := deepcomplete.LspPositionToBytePosition(docInfo.Content, lspPos)
	if posErr != nil {
		logger.Error("Error converting LSP position", "error", posErr)
		return nil, nil // Return nil result for bad position
	}

	// 4. Version Check (Done before calling completer)
	docStoreMutex.RLock()
	currentVersion := docInfo.Version // Use version from initially fetched docInfo
	docStoreMutex.RUnlock()
	// Check against potentially newer version fetched just before call? Less critical now with cancellation.
	// logger.Debug("Version check passed implicitly via context.")

	// 5. Call Core Completer
	logger.Debug("Calling GetCompletionStreamFromFile", "line", line, "col", col, "version", currentVersion)
	var completionBuf bytes.Buffer
	// Note: ctx here is the request-specific context from cancellationMiddleware
	err := completer.GetCompletionStreamFromFile(ctx, absPath, currentVersion, line, col, &completionBuf)

	// 6. Handle Completer Errors
	if err != nil {
		// Cancellation error is handled by cancellationMiddleware, check for others
		if errors.Is(err, context.Canceled) {
			logger.Warn("Completion cancelled during core completer execution (expected if cancellation requested).")
			return nil, &ErrorObject{Code: RequestCancelled, Message: "Request cancelled"}
		} else if errors.Is(err, deepcomplete.ErrOllamaUnavailable) {
			logger.Error("Ollama unavailable for completion", "error", err)
			sendShowMessageNotification(MessageTypeError, "Completion backend (Ollama) is unavailable.") // Notify user
			return nil, &ErrorObject{Code: RequestFailed, Message: "Completion backend unavailable."}
		} else if errors.Is(err, deepcomplete.ErrAnalysisFailed) {
			logger.Warn("Analysis failed non-fatally during completion", "error", err)
			sendShowMessageNotification(MessageTypeWarning, fmt.Sprintf("Analysis issues: %v. Completion may be less accurate.", err))
			// Proceed to format completion if possible, but analysis was partial
		} else {
			// Unexpected internal error
			logger.Error("Error getting completion from core", "error", err)
			return nil, &ErrorObject{Code: RequestFailed, Message: "Completion generation failed internally."}
		}
	}

	// 7. Format Completion Result
	completionText := completionBuf.String()
	completionResult := CompletionList{IsIncomplete: false, Items: []CompletionItem{}} // Default empty
	if completionText != "" {
		// Filtering logic (Cycle 5)
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
			// Format item (existing logic...)
			label := strings.TrimSpace(completionText)
			if firstNewline := strings.Index(label, "\n"); firstNewline != -1 {
				label = label[:firstNewline]
			}
			if len(label) > 50 {
				label = label[:50] + "..."
			}
			kind := CompletionItemKindText // Default
			// ... (kind detection logic) ...
			item := CompletionItem{Label: label, Kind: kind, Detail: "DeepComplete Suggestion"}
			if clientSupportsSnippets {
				item.InsertTextFormat = SnippetFormat
				item.InsertText = completionText + "$0"
			} else {
				item.InsertTextFormat = PlainTextFormat
				item.InsertText = completionText
			}
			completionResult.Items = []CompletionItem{item}
		}
	}

	return completionResult, nil // Return formatted result (or empty list)
}

// hoverHandler handles 'textDocument/hover'.
func hoverHandler(ctx context.Context, event Event, logger *slog.Logger) (result any, errObj *ErrorObject) {
	logger.Info("Core handler: hover")
	var params HoverParams

	// 1. Parse and Validate Parameters
	if event.GetRawParams() == nil {
		return nil, &ErrorObject{Code: InvalidParams, Message: "Missing parameters for hover"}
	}
	if err := json.Unmarshal(event.GetRawParams(), &params); err != nil {
		logger.Error("Failed to unmarshal hover params", "error", err)
		return nil, &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid params for hover: %v", err)}
	}
	// Validate position
	if params.Position.Line < 0 || params.Position.Character < 0 {
		logger.Error("Invalid hover params: negative position", "line", params.Position.Line, "char", params.Position.Character)
		return nil, &ErrorObject{Code: InvalidParams, Message: "Invalid position: line and character must be non-negative."}
	}
	// Validate URI and get path
	absPath, pathErr := deepcomplete.ValidateAndGetFilePath(string(params.TextDocument.URI), logger) // Use string cast
	if pathErr != nil {
		logger.Error("Invalid hover params: bad URI", "uri", params.TextDocument.URI, "error", pathErr)
		return nil, &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid document URI: %v", pathErr)}
	}
	logger = logger.With("uri", string(params.TextDocument.URI), "path", absPath)

	// 2. Get Document State
	docStoreMutex.RLock()
	docInfo, ok := documentStore[params.TextDocument.URI]
	if ok {
		docInfo.LastAccess = time.Now()
	} // Conceptual LRU update
	docStoreMutex.RUnlock()
	if !ok {
		logger.Error("Document not found for hover")
		return nil, nil // Return nil result for missing doc
	}

	// 3. Convert Position
	lspPos := deepcomplete.LSPPosition{Line: params.Position.Line, Character: params.Position.Character}
	line, col, _, posErr := deepcomplete.LspPositionToBytePosition(docInfo.Content, lspPos)
	if posErr != nil {
		logger.Error("Error converting LSP position for hover", "error", posErr)
		return nil, nil // Return nil result for bad position
	}

	// 4. Call Analyzer
	if completer == nil || completer.GetAnalyzer() == nil {
		logger.Error("Analyzer not available for hover request")
		return nil, nil // Return nil result
	}
	analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second) // Use request ctx as parent
	defer cancelAnalysis()
	analysisInfo, analysisErr := completer.GetAnalyzer().Analyze(analysisCtx, absPath, docInfo.Version, line, col)

	if analysisErr != nil && !errors.Is(analysisErr, deepcomplete.ErrAnalysisFailed) {
		logger.Error("Fatal error during analysis/cache check for hover", "error", analysisErr)
		return nil, &ErrorObject{Code: InternalError, Message: "Failed to analyze code for hover."} // Return internal error
	}
	if analysisErr != nil {
		logger.Warn("Analysis for hover completed with errors", "error", analysisErr)
		sendShowMessageNotification(MessageTypeWarning, fmt.Sprintf("Analysis issues during hover: %v", analysisErr))
	}
	if analysisInfo == nil {
		logger.Warn("Analysis for hover returned nil info")
		return nil, nil // Return nil result
	}
	if analysisCtx.Err() != nil {
		logger.Warn("Analysis context cancelled or timed out during hover", "error", analysisCtx.Err())
		// Cancellation middleware handles ctx.Err() == context.Canceled
		return nil, nil // Return nil result for timeout
	}

	// 5. Format Hover Content
	var hoverResult *HoverResult
	if analysisInfo.IdentifierObject != nil {
		obj := analysisInfo.IdentifierObject
		logger.Debug("Found object for hover", "name", obj.Name(), "type", obj.Type().String())

		// Call conceptual helper (implementation needed in deepcomplete_helpers.go)
		hoverContent := formatObjectForHover(obj, analysisInfo, logger)

		if hoverContent != "" {
			hoverResult = &HoverResult{
				Contents: MarkupContent{
					Kind:  MarkupKindMarkdown, // Assume Markdown
					Value: hoverContent,
				},
			}
		} else {
			logger.Debug("Formatter returned empty content for object", "name", obj.Name())
		}
	} else {
		logger.Debug("No specific identifier object found at cursor for hover")
	}

	return hoverResult, nil // Return HoverResult (or nil)
}

// ============================================================================
// Notification Handlers (Executed directly or via simple goroutine)
// ============================================================================

// NOTE: These handlers are currently called directly by the dispatcher.
// They could also be wrapped by a simpler middleware chain if needed (e.g., for panic recovery).

func handleInitializedNotification(ctx context.Context, event InitializedNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'initialized' notification.")
	// Can perform actions after client confirms initialization here
}

func handleExitNotification(ctx context.Context, event ExitNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'exit' notification. Signaling shutdown.")
	select {
	case <-shutdownChan: // Already shutting down
	default:
		close(shutdownChan)
	}
}

func handleDidOpenNotification(ctx context.Context, event DidOpenNotificationEvent, logger *slog.Logger) {
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

	// Validate URI
	absPath, pathErr := deepcomplete.ValidateAndGetFilePath(string(params.TextDocument.URI), logger) // Use string cast
	if pathErr != nil {
		logger.Error("Invalid URI in didOpen", "uri", params.TextDocument.URI, "error", pathErr)
		return
	}
	// Validate version
	if params.TextDocument.Version < 0 {
		logger.Error("Invalid negative version in didOpen", "version", params.TextDocument.Version, "uri", params.TextDocument.URI)
		return
	}
	logger = logger.With("uri", string(params.TextDocument.URI), "path", absPath)

	docStoreMutex.Lock()
	// --- Eviction Logic (Conceptual - needs refinement/LRU implementation) ---
	if len(documentStore) >= maxOpenDocuments {
		oldestURI := ""
		oldestTime := time.Now()
		for uri, docInfo := range documentStore {
			if docInfo.LastAccess.Before(oldestTime) {
				oldestTime = docInfo.LastAccess
				oldestURI = string(uri)
			}
		}
		if oldestURI != "" {
			evictedURI := DocumentURI(oldestURI)
			logger.Info("Max open documents reached, evicting least recently used", "limit", maxOpenDocuments, "evicted_uri", evictedURI)
			delete(documentStore, evictedURI)
			docsEvicted.Add(1)
			// Cancel requests & invalidate caches for evicted URI
			go cancelActiveRequestsForURI(evictedURI, logger)
			if completer != nil {
				go completer.InvalidateMemoryCacheForURI(string(evictedURI), -1)
			} // Use string cast
			// Need dir for bbolt invalidation - maybe skip on eviction?
		} else {
			logger.Warn("Document store full but failed to find LRU entry to evict.")
		}
	}
	// --- End Eviction Logic ---

	// Add the new document
	documentStore[params.TextDocument.URI] = VersionedContent{
		Content:    []byte(params.TextDocument.Text),
		Version:    params.TextDocument.Version,
		LastAccess: time.Now(), // Set initial access time
	}
	docStoreMutex.Unlock()

	logger.Info("Opened and stored document", "version", params.TextDocument.Version, "language", params.TextDocument.LanguageID, "length", len(params.TextDocument.Text))
}

func handleDidCloseNotification(ctx context.Context, event DidCloseNotificationEvent, logger *slog.Logger) {
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

	// Validate URI (optional but good practice)
	if _, pathErr := deepcomplete.ValidateAndGetFilePath(string(params.TextDocument.URI), logger); pathErr != nil { // Use string cast
		logger.Error("Invalid URI in didClose", "uri", params.TextDocument.URI, "error", pathErr)
		return
	}
	logger = logger.With("uri", string(params.TextDocument.URI))

	docStoreMutex.Lock()
	delete(documentStore, params.TextDocument.URI)
	docStoreMutex.Unlock()

	logger.Info("Closed and removed document")

	// Invalidate memory cache for closed document
	if completer != nil {
		uriStr := string(params.TextDocument.URI) // Use string for the call
		go func(uri string) {
			// Create new logger for goroutine context
			closeLogger := slog.Default().With("uri", uri)
			closeLogger.Debug("Attempting memory cache invalidation on didClose")
			if err := completer.InvalidateMemoryCacheForURI(uri, -1); err != nil { // Pass string
				closeLogger.Error("Error invalidating memory cache after didClose", "error", err)
			} else {
				closeLogger.Debug("Memory cache invalidated due to didClose.")
			}
		}(uriStr)
	}
}

func handleDidChangeNotification(ctx context.Context, event DidChangeNotificationEvent, logger *slog.Logger) {
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

	// Validate URI
	absPath, pathErr := deepcomplete.ValidateAndGetFilePath(string(params.TextDocument.URI), logger) // Use string cast
	if pathErr != nil {
		logger.Error("Invalid URI in didChange", "uri", params.TextDocument.URI, "error", pathErr)
		return
	}
	// Validate version
	if params.TextDocument.Version < 0 {
		logger.Error("Invalid negative version in didChange", "version", params.TextDocument.Version, "uri", params.TextDocument.URI)
		return
	}
	logger = logger.With("uri", string(params.TextDocument.URI), "path", absPath, "newVersion", params.TextDocument.Version)

	// Assuming TextDocumentSyncKindFull
	if len(params.ContentChanges) == 0 {
		logger.Warn("Received didChange notification with no content changes.")
		return
	}
	newText := params.ContentChanges[len(params.ContentChanges)-1].Text

	docStoreMutex.Lock()
	// --- Stricter Version Check ---
	existing, ok := documentStore[params.TextDocument.URI]
	shouldUpdate := !ok || params.TextDocument.Version > existing.Version
	if !shouldUpdate {
		if ok {
			logger.Warn("Received out-of-order or redundant document change notification. Ignoring.", "receivedVersion", params.TextDocument.Version, "storedVersion", existing.Version)
		} else {
			logger.Warn("Ignoring didChange notification for unknown reason (shouldUpdate=false, ok=false).", "receivedVersion", params.TextDocument.Version)
		}
		docStoreMutex.Unlock()
		return
	}
	// --- End Stricter Version Check ---

	// Proceed with update
	documentStore[params.TextDocument.URI] = VersionedContent{
		Content:    []byte(newText),
		Version:    params.TextDocument.Version,
		LastAccess: time.Now(), // Update access time on change
	}
	docStoreMutex.Unlock()

	logger.Info("Updated document", "newLength", len(newText))

	// Cancel ongoing requests for this document
	cancelActiveRequestsForURI(params.TextDocument.URI, logger)

	// Invalidate caches
	if completer != nil {
		uri := params.TextDocument.URI
		uriStr := string(uri) // String version for calls needing string
		version := params.TextDocument.Version
		dir := filepath.Dir(absPath) // Use validated absolute path

		// Invalidate Bbolt cache (based on directory)
		go func(d string) {
			bboltLogger := slog.Default().With("uri", uriStr, "dir", d)
			bboltLogger.Debug("Attempting bbolt cache invalidation")
			if err := completer.InvalidateAnalyzerCache(d); err != nil {
				bboltLogger.Error("Error invalidating bbolt cache after didChange", "error", err)
			} else {
				bboltLogger.Debug("Bbolt cache invalidated due to didChange.")
			}
		}(dir)

		// Invalidate Memory cache (based on URI and potentially version)
		go func(u string, v int) { // Pass string URI
			memLogger := slog.Default().With("uri", u, "version", v)
			memLogger.Debug("Attempting memory cache invalidation")
			if err := completer.InvalidateMemoryCacheForURI(u, v); err != nil { // Call with string
				memLogger.Error("Error invalidating memory cache after didChange", "error", err)
			} else {
				memLogger.Debug("Memory cache invalidated due to didChange.")
			}
		}(uriStr, version)
	}
}

func handleDidChangeConfigurationNotification(ctx context.Context, event DidChangeConfigurationNotificationEvent, logger *slog.Logger) {
	logger.Info("Handling 'workspace/didChangeConfiguration' notification...")
	settingsRaw := event.GetRawParams()
	if settingsRaw == nil {
		logger.Warn("Received didChangeConfiguration with nil settings.")
		return
	}
	logger.Debug("Received configuration change notification", "raw_settings", string(settingsRaw))

	if completer == nil {
		logger.Warn("Cannot apply configuration changes, completer not initialized.")
		return
	}

	newConfig := completer.GetCurrentConfig() // Get a copy
	settingsGroup := gjson.ParseBytes(settingsRaw)
	configPrefix := "deepcomplete." // Assuming settings are nested under this key
	configUpdated := false

	updateIfChanged := func(key string, updateFunc func(val gjson.Result)) {
		result := settingsGroup.Get(configPrefix + key)
		if result.Exists() {
			updateFunc(result)
			configUpdated = true
			logger.Debug("Config key found in notification", "key", configPrefix+key, "value", result.Value())
		}
	}

	// Update fields using the helper
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
		} else {
			logger.Warn("Config 'stop' value is not an array, ignoring change.", "key", configPrefix+"stop", "value", val.Raw)
		}
	})

	if configUpdated {
		// Validate the potentially modified config *before* applying
		if err := newConfig.Validate(); err != nil {
			logger.Error("Updated configuration is invalid, not applying.", "error", err)
			sendShowMessageNotification(MessageTypeError, fmt.Sprintf("Invalid deepcomplete configuration: %v", err)) // Notify user
			return
		}
		// Apply the validated config
		if err := completer.UpdateConfig(newConfig); err != nil {
			logger.Error("Error applying updated configuration", "error", err)
			sendShowMessageNotification(MessageTypeError, fmt.Sprintf("Failed to apply configuration: %v", err)) // Notify user
		} else {
			logger.Info("Successfully applied updated configuration from client.")
			// Optionally send info message: sendShowMessageNotification(MessageTypeInfo, "DeepComplete configuration updated.")
		}
	} else {
		logger.Info("No relevant configuration changes detected in notification.")
	}
}

func handleCancelRequestNotification(ctx context.Context, event CancelRequestEvent, logger *slog.Logger) {
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
	reqID := idResult.Value() // gjson handles number/string ID correctly

	logger = logger.With("cancelledRequestID", reqID)
	logger.Info("Received cancellation request")

	activeRequestMut.Lock()
	cancelFunc, ok := activeRequests[reqID]
	delete(activeRequests, reqID) // Remove entry regardless
	activeRequestMut.Unlock()

	if ok && cancelFunc != nil { // Defensive: Check cancelFunc is not nil
		cancelFunc() // Call the context cancellation function
		logger.Info("Cancelled active request")
	} else {
		logger.Warn("No active request found to cancel (already finished or invalid ID)")
	}
}

// ============================================================================
// Request Cancellation Helpers
// ============================================================================

func registerRequest(id any, cancel context.CancelFunc, logger *slog.Logger) {
	if id == nil {
		logger.Warn("Attempted to register request with nil ID for cancellation.")
		return
	}
	if cancel == nil {
		logger.Error("Attempted to register request with nil cancel function", "id", id)
		return
	}
	activeRequestMut.Lock()
	defer activeRequestMut.Unlock()
	if _, exists := activeRequests[id]; exists {
		logger.Warn("Request ID already registered for cancellation, overwriting.", "id", id)
	}
	activeRequests[id] = cancel
	logger.Debug("Registered request for cancellation", "id", id)
}

func unregisterRequest(id any, logger *slog.Logger) {
	if id == nil {
		return
	}
	activeRequestMut.Lock()
	defer activeRequestMut.Unlock()
	delete(activeRequests, id) // Delete is safe even if key doesn't exist
	logger.Debug("Unregistered request", "id", id)
}

// cancelActiveRequestsForURI attempts to cancel requests associated with a URI.
// NOTE: This is inefficient without a proper mapping.
func cancelActiveRequestsForURI(uri DocumentURI, logger *slog.Logger) {
	// TODO: Implement efficient cancellation by URI.
	// Requires associating request IDs with URIs during registration.
	logger.Warn("Attempting cancellation by URI (currently ineffective)", "uri", uri)
	// Conceptual future implementation:
	// activeRequestMut.Lock()
	// idsToCancel := findRequestIDsForURI(uri) // Needs a new map[DocumentURI][]requestID
	// for _, id := range idsToCancel { ... cancel ... delete ... }
	// activeRequestMut.Unlock()
	// removeURIFromMapping(uri)
}

// ============================================================================
// Notification Sending Helper
// ============================================================================

// sendShowMessageNotification sends a window/showMessage notification to the client.
func sendShowMessageNotification(level MessageType, message string) {
	// Use default logger for this helper
	logger := slog.Default()
	if notificationQueue == nil {
		logger.Error("Cannot send notification, queue not initialized", "level", level, "message", message)
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

	// Use non-blocking send with timeout
	select {
	case notificationQueue <- notification:
		logger.Debug("Sent showMessage notification to queue", "level", level, "message", message)
	case <-time.After(notifySendTimeout): // Use specific timeout for notifications
		notificationQueueTimeout.Add(1)
		logger.Warn("Timeout sending showMessage notification to queue", "message", message, "timeout", notifySendTimeout)
		// No check for serverCtx.Done() here? If server is shutting down, notifications might be less critical.
	}
}

// ============================================================================
// I/O Helpers
// ============================================================================

// readMessage reads a single JSON-RPC message based on Content-Length header.
func readMessage(reader *bufio.Reader) ([]byte, error) {
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
		} // End of headers

		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			valueStr := strings.TrimSpace(line[len("content-length:"):])
			_, err := fmt.Sscan(valueStr, &contentLength) // Use Sscan for robustness
			if err != nil {
				slog.Warn("Failed to parse Content-Length header value", "header_line", line, "value_part", valueStr, "error", err)
				contentLength = -1 // Mark as invalid
			}
		}
		// Ignore other headers like Content-Type for now
	}

	if contentLength < 0 {
		return nil, fmt.Errorf("missing or invalid Content-Length header found")
	}
	if contentLength == 0 {
		slog.Debug("Received message with Content-Length: 0")
		return []byte{}, nil
	}
	// Defensive: Add max size check?
	// const maxMessageSize = 10 * 1024 * 1024 // 10MB
	// if contentLength > maxMessageSize { ... return error ... }

	body := make([]byte, contentLength)
	n, err := io.ReadFull(reader, body) // Read exactly contentLength bytes
	if err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			slog.Warn("Client disconnected before sending full message body", "expected", contentLength, "read", n)
			return nil, io.EOF
		}
		return nil, fmt.Errorf("failed reading message body (read %d/%d bytes): %w", n, contentLength, err)
	}
	return body, nil
}

// writeMessage encodes a message (Response or Notification) and writes it.
func writeMessage(writer io.Writer, message interface{}) error {
	// Defensive: Check for nil message
	if message == nil {
		slog.Error("Attempted to write nil message")
		return errors.New("cannot write nil message")
	}

	content, err := json.Marshal(message)
	if err != nil {
		slog.Error("Error marshalling message for output", "error", err, "messageType", fmt.Sprintf("%T", message))
		// Attempt to marshal a generic internal error response ONLY if original was ResponseMessage
		if respMsg, ok := message.(ResponseMessage); ok {
			errMsg := ResponseMessage{
				JSONRPC: "2.0",
				ID:      respMsg.ID, // Use original ID if possible
				Error:   &ErrorObject{Code: InternalError, Message: "Failed to marshal server response"},
			}
			content, err = json.Marshal(errMsg) // Try marshalling error message
			if err != nil {
				slog.Error("Failed to marshal generic error response", "error", err)
				return fmt.Errorf("failed to marshal even generic error response: %w", err)
			}
		} else {
			// Don't try to send an error response for a failed notification marshal
			return fmt.Errorf("failed to marshal notification message: %w", err)
		}
	}

	// Use standard CRLF line endings for headers
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(content))
	slog.Debug("Sending message", "header", header, "content_length", len(content), "content_preview", firstN(string(content), 100))

	// Write header
	if _, err := writer.Write([]byte(header)); err != nil {
		slog.Error("Error writing message header", "error", err)
		return fmt.Errorf("error writing header: %w", err)
	}
	// Then write content
	if _, err := writer.Write(content); err != nil {
		slog.Error("Error writing message content", "error", err)
		return fmt.Errorf("error writing content: %w", err)
	}
	// Flushing is generally handled by the OS buffer or the client reading stdio,
	// explicitly flushing might be needed only for specific writers.
	return nil
}

// Helper function to truncate strings for logging
func firstN(s string, n int) string {
	if n < 0 {
		n = 0
	} // Ensure non-negative length
	// Iterate by rune, not byte, to handle multi-byte characters correctly
	count := 0
	for i := range s {
		if count == n {
			return s[:i] + "..." // Return substring up to byte index i
		}
		count++
	}
	// String is shorter than or equal to n
	return s
}

// Conceptual helper (implementation needed in deepcomplete_helpers.go)
func formatObjectForHover(obj types.Object, analysisInfo *deepcomplete.AstContextInfo, logger *slog.Logger) string {
	// Placeholder - Requires implementation in helpers file using analysisInfo
	logger.Warn("formatObjectForHover called but using placeholder implementation")
	if obj != nil {
		// Basic formatting without doc comments or proper qualifier
		qualifier := func(other *types.Package) string { return "" } // Dummy qualifier
		definition := types.ObjectString(obj, qualifier)
		return fmt.Sprintf("```go\n%s\n```\n\n(Documentation lookup not implemented)", definition)
	}
	return ""
}
