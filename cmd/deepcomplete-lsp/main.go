package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	// NOTE: Replace with your actual module path
	"github.com/shehackedyou/deepcomplete"
)

// ============================================================================
// Global Variables & Constants
// ============================================================================

var (
	// completer is initialized in main and used by handlers.
	completer *deepcomplete.DeepCompleter

	// documentStore holds the content of open documents, keyed by URI.
	documentStore = make(map[DocumentURI][]byte)
	docStoreMutex sync.RWMutex // Protects concurrent access to documentStore.

	// clientSupportsSnippets tracks if the connected client supports snippet format. (Set during initialize).
	clientSupportsSnippets bool

	// --- Event-Driven Architecture Components (Cycle 1, Step 1) ---

	// eventQueue holds incoming LSP messages parsed into Event structs.
	eventQueue chan Event
	// responseQueue holds results/errors from processed requests waiting to be written.
	responseQueue chan ResponseWorkItem

	// Default buffer size for channels.
	defaultQueueSize = 100
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
// Event Definitions (Cycle 1, Step 1)
// ============================================================================

// Event represents a parsed LSP message (request or notification).
type Event interface {
	GetType() string   // Returns the LSP method name.
	GetRequestID() any // Returns the request ID (nil for notifications).
}

// BaseEvent provides common fields for events.
type BaseEvent struct {
	Type      string
	RequestID any // nil for notifications
}

func (e BaseEvent) GetType() string {
	return e.Type
}
func (e BaseEvent) GetRequestID() any {
	return e.RequestID
}

// InitializeRequestEvent represents the 'initialize' request.
type InitializeRequestEvent struct {
	BaseEvent
	Params InitializeParams
}

// ShutdownRequestEvent represents the 'shutdown' request.
type ShutdownRequestEvent struct {
	BaseEvent
}

// CompletionRequestEvent represents the 'textDocument/completion' request.
type CompletionRequestEvent struct {
	BaseEvent
	Params CompletionParams
}

// DidOpenNotificationEvent represents the 'textDocument/didOpen' notification.
type DidOpenNotificationEvent struct {
	BaseEvent
	Params DidOpenTextDocumentParams
}

// DidCloseNotificationEvent represents the 'textDocument/didClose' notification.
type DidCloseNotificationEvent struct {
	BaseEvent
	Params DidCloseTextDocumentParams
}

// DidChangeNotificationEvent represents the 'textDocument/didChange' notification.
type DidChangeNotificationEvent struct {
	BaseEvent
	Params DidChangeTextDocumentParams
}

// DidChangeConfigurationNotificationEvent represents the 'workspace/didChangeConfiguration' notification.
type DidChangeConfigurationNotificationEvent struct {
	BaseEvent
	Params DidChangeConfigurationParams
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
	Params CancelParams
}

// UnknownEvent represents an unhandled or unparseable message.
type UnknownEvent struct {
	BaseEvent
	RawMessage json.RawMessage
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

type RequestMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"` // Use 'any' for flexibility (number or string)
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type ResponseMessage struct {
	JSONRPC string       `json:"jsonrpc"`
	ID      any          `json:"id,omitempty"`
	Result  any          `json:"result,omitempty"`
	Error   *ErrorObject `json:"error,omitempty"`
}

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
// These are simplified versions. A full implementation might use an LSP library.

type DocumentURI string

type Position struct {
	Line      uint32 `json:"line"`      // 0-based
	Character uint32 `json:"character"` // 0-based, UTF-16 offset
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

// ClientCapabilities represents the capabilities provided by the LSP client.
type ClientCapabilities struct {
	Workspace    *WorkspaceClientCapabilities    `json:"workspace,omitempty"`
	TextDocument *TextDocumentClientCapabilities `json:"textDocument,omitempty"`
}
type WorkspaceClientCapabilities struct {
	Configuration bool `json:"configuration,omitempty"` // Client supports workspace/configuration requests.
}
type TextDocumentClientCapabilities struct {
	Completion *CompletionClientCapabilities `json:"completion,omitempty"`
}
type CompletionClientCapabilities struct {
	CompletionItem *CompletionItemClientCapabilities `json:"completionItem,omitempty"`
}
type CompletionItemClientCapabilities struct {
	SnippetSupport bool `json:"snippetSupport,omitempty"` // Client supports snippet formats.
}

type InitializeResult struct {
	Capabilities ServerCapabilities `json:"capabilities"`
	ServerInfo   *ServerInfo        `json:"serverInfo,omitempty"`
}

// ServerCapabilities defines the capabilities provided by this LSP server.
type ServerCapabilities struct {
	TextDocumentSync   *TextDocumentSyncOptions `json:"textDocumentSync,omitempty"`   // How documents are synced.
	CompletionProvider *CompletionOptions       `json:"completionProvider,omitempty"` // Advertise completion support.
}

type TextDocumentSyncOptions struct {
	OpenClose bool                 `json:"openClose,omitempty"` // Supports didOpen/didClose notifications.
	Change    TextDocumentSyncKind `json:"change,omitempty"`    // Specifies how changes are synced.
}

// TextDocumentSyncKind defines the type of text document synchronization.
type TextDocumentSyncKind int

const (
	TextDocumentSyncKindNone TextDocumentSyncKind = 0 // Not synced.
	TextDocumentSyncKindFull TextDocumentSyncKind = 1 // Synced by sending full content.
	// TextDocumentSyncKindIncremental TextDocumentSyncKind = 2 // Incremental changes (not implemented yet).
)

type CompletionOptions struct {
	// TriggerCharacters []string `json:"triggerCharacters,omitempty"`
	// ResolveProvider bool `json:"resolveProvider,omitempty"`
}

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
	Version int `json:"version"` // Document version after the change.
}

// TextDocumentContentChangeEvent represents a change to a text document.
// For full sync, this contains the entire new text content.
type TextDocumentContentChangeEvent struct {
	Text string `json:"text"` // The new full content for full sync.
}

type DidChangeConfigurationParams struct {
	Settings json.RawMessage `json:"settings"` // The new configuration settings. Structure depends on client.
}

type CompletionParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     Position               `json:"position"`
	Context      *CompletionContext     `json:"context,omitempty"` // Information about why completion was triggered.
}

type CompletionContext struct {
	TriggerKind      CompletionTriggerKind `json:"triggerKind"`
	TriggerCharacter string                `json:"triggerCharacter,omitempty"`
}

// CompletionTriggerKind indicates how completion was triggered.
type CompletionTriggerKind int

const (
	CompletionTriggerKindInvoked              CompletionTriggerKind = 1 // Explicitly invoked (e.g., Ctrl+Space).
	CompletionTriggerKindTriggerChar          CompletionTriggerKind = 2 // Triggered by typing a trigger character.
	CompletionTriggerKindTriggerForIncomplete CompletionTriggerKind = 3 // Retriggered completion on existing list.
)

type CompletionList struct {
	IsIncomplete bool             `json:"isIncomplete"` // True if list is potentially incomplete.
	Items        []CompletionItem `json:"items"`
}

type CompletionItem struct {
	Label            string             `json:"label"`                      // Text shown in the UI list.
	Kind             CompletionItemKind `json:"kind,omitempty"`             // Type of completion item (e.g., Function, Variable).
	Detail           string             `json:"detail,omitempty"`           // Additional info shown in the UI.
	Documentation    string             `json:"documentation,omitempty"`    // Documentation shown on hover/selection.
	InsertTextFormat InsertTextFormat   `json:"insertTextFormat,omitempty"` // Format of the InsertText (PlainText or Snippet).
	InsertText       string             `json:"insertText,omitempty"`       // The text/snippet to insert.
}

// CompletionItemKind represents the type of a completion item.
type CompletionItemKind int // Simplified, see LSP spec for full list

const (
	CompletionItemKindText     CompletionItemKind = 1
	CompletionItemKindFunction CompletionItemKind = 3
	CompletionItemKindVariable CompletionItemKind = 6
	CompletionItemKindKeyword  CompletionItemKind = 14
	// Add more kinds as needed...
)

// InsertTextFormat indicates whether InsertText is plain text or a snippet.
type InsertTextFormat int

const (
	PlainTextFormat InsertTextFormat = 1
	SnippetFormat   InsertTextFormat = 2 // Supports placeholders like $1, $0.
)

// CancelParams defines the parameters for the '$/cancelRequest' notification.
type CancelParams struct {
	ID any `json:"id"` // ID of the request to cancel.
}

// ============================================================================
// Main Server Logic
// ============================================================================

func main() {
	// Setup logging to a file.
	logFile, err := os.OpenFile("deepcomplete-lsp.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0660)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}
	defer logFile.Close()
	log.SetOutput(logFile)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	log.Println("DeepComplete LSP server starting...")

	// Initialize the core DeepCompleter service.
	var initErr error
	completer, initErr = deepcomplete.NewDeepCompleter()
	if initErr != nil {
		log.Fatalf("Failed to initialize DeepCompleter service: %v", initErr)
	}
	defer func() {
		log.Println("Closing DeepCompleter service...")
		if err := completer.Close(); err != nil {
			log.Printf("Error closing completer: %v", err)
		}
	}()
	log.Println("DeepCompleter service initialized.")

	// --- Initialize Event Queues (Cycle 1, Step 1) ---
	eventQueue = make(chan Event, defaultQueueSize)
	responseQueue = make(chan ResponseWorkItem, defaultQueueSize)
	log.Printf("Initialized event and response queues (size: %d)", defaultQueueSize)

	// TODO (Cycle 1): Start Reader, Dispatcher, Writer goroutines here.

	// --- Current Synchronous Message Processing Loop (To be replaced in Cycle 1) ---
	reader := bufio.NewReader(os.Stdin)
	writer := os.Stdout

	// Main message processing loop.
	for {
		var panicErr error
		func() {
			// Recover from panics within message handling to keep server running.
			defer func() {
				if r := recover(); r != nil {
					panicErr = fmt.Errorf("lsp handler panicked: %v\n%s", r, string(debug.Stack()))
					log.Printf("PANIC: %v", panicErr)
				}
			}()

			content, err := readMessage(reader)
			if err != nil {
				if err == io.EOF {
					log.Println("Client closed connection (EOF). Exiting.")
					// TODO (Cycle 1): Need graceful shutdown mechanism for goroutines.
					os.Exit(0)
				}
				log.Printf("Error reading message: %v", err)
				// In event-driven model, reader should continue trying to read.
				// Error handling might involve sending an error response if possible.
				return // Continue loop after read error (for now).
			}
			log.Printf("Received message: %s", string(content))

			var baseMessage RequestMessage // Using RequestMessage to easily get ID/Method first
			if err := json.Unmarshal(content, &baseMessage); err != nil {
				log.Printf("Error decoding base JSON message: %v", err)
				// Cannot send error response without ID.
				// TODO (Cycle 1): Decide how to handle parse errors in event model.
				return
			}

			// TODO (Cycle 1): Instead of handling here, parse into specific Event struct
			//                and send to eventQueue.
			// Example (Conceptual):
			// event := parseMessageToEvent(content, baseMessage)
			// eventQueue <- event

			// --- Existing Synchronous Handling (To be removed/refactored) ---
			ctx := context.Background() // Context needs better management in event model
			if baseMessage.ID != nil {
				handleRequest(ctx, baseMessage, writer) // Will be handled by workers
			} else {
				handleNotification(ctx, baseMessage) // Will be handled by dispatcher/quick workers
			}
			// --- End Existing Synchronous Handling ---

		}() // End of deferred recovery function call

		if panicErr != nil {
			log.Println("Continuing after recovered panic.")
			// TODO (Cycle 1): Consider how panics in workers/dispatcher are handled.
		}
	}
}

// readMessage reads a single JSON-RPC message based on Content-Length header.
func readMessage(reader *bufio.Reader) ([]byte, error) {
	var contentLength int = -1
	for { // Read headers.
		line, err := reader.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("failed reading header line: %w", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		} // End of headers.
		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			_, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength)
			if err != nil {
				log.Printf("Warning: Failed to parse Content-Length header '%s': %v", line, err)
			}
		}
	}
	if contentLength < 0 {
		return nil, fmt.Errorf("missing or invalid Content-Length header")
	}
	if contentLength == 0 {
		return []byte{}, nil
	} // Valid empty message.

	body := make([]byte, contentLength)
	n, err := io.ReadFull(reader, body) // Read exact body length.
	if err != nil {
		return nil, fmt.Errorf("failed reading body (read %d/%d bytes): %w", n, contentLength, err)
	}
	return body, nil
}

// writeMessage encodes a message to JSON and writes it to the writer with headers.
// TODO (Cycle 1): This will be used by the Response Writer goroutine.
func writeMessage(writer io.Writer, message interface{}) error {
	content, err := json.Marshal(message)
	if err != nil {
		// Log critical error as this prevents sending a response/error back.
		log.Printf("CRITICAL: Error marshalling message: %v (Message Type: %T)", err, message)
		// Don't return error here, as the writer goroutine should continue trying to write other messages.
		// Consider alternative error reporting if marshalling fails consistently.
		return fmt.Errorf("error marshalling message: %w", err) // Or maybe just log and continue?
	}
	log.Printf("Sending message: %s", string(content))
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(content))
	// Use a mutex if writer is shared (os.Stdout is generally safe for concurrent writes, but explicit locking is safer)
	// For now, assume a single writer goroutine owns stdout.
	if _, err := writer.Write([]byte(header)); err != nil {
		log.Printf("Error writing message header: %v", err)
		return fmt.Errorf("error writing header: %w", err)
	}
	if _, err := writer.Write(content); err != nil {
		log.Printf("Error writing message content: %v", err)
		return fmt.Errorf("error writing content: %w", err)
	}
	return nil
}

// ----------------------------------------------------------------------------
// --- Synchronous Handlers (To be refactored/moved into event handlers/workers in Cycle 1) ---
// ----------------------------------------------------------------------------

// handleRequest processes incoming requests and sends responses.
// TODO (Cycle 1): This logic will move into worker goroutines.
func handleRequest(ctx context.Context, req RequestMessage, writer io.Writer) {
	response := ResponseMessage{JSONRPC: "2.0", ID: req.ID}

	switch req.Method {
	case "initialize":
		log.Println("Handling 'initialize' request...")
		var params InitializeParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			response.Error = &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid params for initialize: %v", err)}
		} else {
			if params.ClientInfo != nil {
				log.Printf("Client Info: Name=%s, Version=%s", params.ClientInfo.Name, params.ClientInfo.Version)
			}
			// Check client capabilities (Cycle 10).
			clientSupportsSnippets = false // Default
			if params.Capabilities.TextDocument != nil &&
				params.Capabilities.TextDocument.Completion != nil &&
				params.Capabilities.TextDocument.Completion.CompletionItem != nil &&
				params.Capabilities.TextDocument.Completion.CompletionItem.SnippetSupport {
				clientSupportsSnippets = true
				log.Println("Client supports snippets.")
			} else {
				log.Println("Client does not support snippets.")
			}
			// Advertise server capabilities.
			response.Result = InitializeResult{
				Capabilities: ServerCapabilities{
					TextDocumentSync:   &TextDocumentSyncOptions{OpenClose: true, Change: TextDocumentSyncKindFull},
					CompletionProvider: &CompletionOptions{},
				},
				ServerInfo: &ServerInfo{Name: "DeepComplete LSP", Version: "0.0.1" /* TODO: Version */},
			}
		}

	case "shutdown":
		log.Println("Handling 'shutdown' request...")
		response.Result = nil // Success.
		// TODO (Cycle 1): Initiate graceful shutdown of goroutines.

	case "textDocument/completion":
		log.Println("Handling 'textDocument/completion' request...")
		var params CompletionParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			response.Error = &ErrorObject{Code: InvalidParams, Message: fmt.Sprintf("Invalid params for completion: %v", err)}
			break
		}
		if params.Context != nil {
			log.Printf("Completion Context: TriggerKind=%d, TriggerChar=%q", params.Context.TriggerKind, params.Context.TriggerCharacter)
		}

		docStoreMutex.RLock()
		contentBytes, ok := documentStore[params.TextDocument.URI]
		docStoreMutex.RUnlock()
		if !ok {
			log.Printf("Error: Document not found for completion: %s", params.TextDocument.URI)
			response.Error = &ErrorObject{Code: RequestFailed, Message: fmt.Sprintf("Document not found: %s", params.TextDocument.URI)}
			break
		}

		// TODO (Cycle 1): This conversion and the completer call below should happen in a worker goroutine.
		line, col, _, err := deepcomplete.LspPositionToBytePosition(contentBytes, params.Position)
		if err != nil {
			log.Printf("Error converting LSP position for %s: %v", params.TextDocument.URI, err)
			response.Error = &ErrorObject{Code: RequestFailed, Message: fmt.Sprintf("Failed to convert position: %v", err)}
			break
		}

		// Call core completer.
		var completionBuf bytes.Buffer
		// Context needs to be managed per-request, potentially cancellable.
		completionCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
		defer cancel()
		err = completer.GetCompletionStreamFromFile(completionCtx, string(params.TextDocument.URI), line, col, &completionBuf)
		if err != nil {
			log.Printf("Error getting completion from core for %s (%d:%d): %v", params.TextDocument.URI, line, col, err)
			response.Error = &ErrorObject{Code: RequestFailed, Message: fmt.Sprintf("Completion failed: %v", err)}
			break
		}

		completionText := completionBuf.String()
		if completionText == "" {
			response.Result = CompletionList{IsIncomplete: false, Items: []CompletionItem{}}
			break
		}

		// Format completion item.
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
		// Set insert text based on client snippet support (Cycle 10).
		if clientSupportsSnippets {
			item.InsertTextFormat = SnippetFormat
			item.InsertText = completionText + "$0" // Add final cursor stop.
		} else {
			item.InsertTextFormat = PlainTextFormat
			item.InsertText = completionText
		}
		response.Result = CompletionList{IsIncomplete: false, Items: []CompletionItem{item}}

	default:
		log.Printf("Received unhandled request method: %s", req.Method)
		response.Error = &ErrorObject{Code: MethodNotFound, Message: fmt.Sprintf("Method not found: %s", req.Method)}
	}

	// TODO (Cycle 1): This write needs to happen via the responseQueue and Writer goroutine.
	if err := writeMessage(writer, response); err != nil {
		log.Printf("Error sending response for ID %v: %v", req.ID, err)
	}
}

// handleNotification processes incoming notifications.
// TODO (Cycle 1): This logic will move into the dispatcher or quick worker goroutines.
func handleNotification(ctx context.Context, notif NotificationMessage) {
	switch notif.Method {
	case "initialized":
		log.Println("Received 'initialized' notification from client.")
		// No action needed, just log.

	case "exit":
		log.Println("Handling 'exit' notification. Server exiting.")
		// TODO (Cycle 1): Initiate graceful shutdown of all goroutines.
		os.Exit(0) // Assume clean exit for now.

	case "textDocument/didOpen":
		log.Println("Handling 'textDocument/didOpen' notification...")
		var params DidOpenTextDocumentParams
		if err := json.Unmarshal(notif.Params, &params); err != nil {
			log.Printf("Error decoding didOpen params: %v", err)
			return
		}
		// This state update needs to be thread-safe.
		docStoreMutex.Lock()
		documentStore[params.TextDocument.URI] = []byte(params.TextDocument.Text)
		docStoreMutex.Unlock()
		log.Printf("Opened and stored document: %s (Version: %d, Language: %s, Length: %d)", params.TextDocument.URI, params.TextDocument.Version, params.TextDocument.LanguageID, len(params.TextDocument.Text))
		// TODO (Optional): Trigger background analysis?

	case "textDocument/didClose":
		log.Println("Handling 'textDocument/didClose' notification...")
		var params DidCloseTextDocumentParams
		if err := json.Unmarshal(notif.Params, &params); err != nil {
			log.Printf("Error decoding didClose params: %v", err)
			return
		}
		// This state update needs to be thread-safe.
		docStoreMutex.Lock()
		delete(documentStore, params.TextDocument.URI)
		docStoreMutex.Unlock()
		log.Printf("Closed and removed document: %s", params.TextDocument.URI)
		// TODO: Cancel any ongoing analysis/completion for this document?

	case "textDocument/didChange":
		log.Println("Handling 'textDocument/didChange' notification...")
		var params DidChangeTextDocumentParams
		if err := json.Unmarshal(notif.Params, &params); err != nil {
			log.Printf("Error decoding didChange params: %v", err)
			return
		}
		// Assuming full sync.
		if len(params.ContentChanges) != 1 {
			log.Printf("Warning: Expected exactly one content change for full sync, got %d for %s. Using first change.", len(params.ContentChanges), params.TextDocument.URI)
			if len(params.ContentChanges) == 0 {
				return
			}
		}
		newText := params.ContentChanges[0].Text
		// This state update needs to be thread-safe.
		docStoreMutex.Lock()
		documentStore[params.TextDocument.URI] = []byte(newText)
		docStoreMutex.Unlock()
		log.Printf("Updated document via didChange (full sync): %s (Version: %d, New Length: %d)", params.TextDocument.URI, params.TextDocument.Version, len(newText))
		// TODO (Optional): Trigger background analysis? Invalidate cache?

	case "workspace/didChangeConfiguration":
		log.Println("Handling 'workspace/didChangeConfiguration' notification...")
		var params DidChangeConfigurationParams
		if err := json.Unmarshal(notif.Params, &params); err != nil {
			log.Printf("Error decoding didChangeConfiguration params: %v", err)
			return
		}
		log.Printf("Received configuration change notification. Raw settings: %s", string(params.Settings))

		// TODO (Cycle 4): Use gjson for selective parsing here.
		var fileCfg deepcomplete.FileConfig
		if err := json.Unmarshal(params.Settings, &fileCfg); err != nil {
			log.Printf("Warning: Could not unmarshal settings into FileConfig structure: %v. Config not updated.", err)
			return
		}

		if completer != nil {
			// Merge new settings with defaults (simple approach).
			// This merging logic needs to be thread-safe if config can be read concurrently.
			// completer.UpdateConfig handles locking internally.
			mergedConfig := deepcomplete.DefaultConfig // Start with defaults
			// Apply existing config from completer first (thread-safe read needed if not using UpdateConfig)
			// currentCfg := completer.GetCurrentConfig() // Hypothetical thread-safe getter
			// mergedConfig = currentCfg
			if fileCfg.OllamaURL != nil {
				mergedConfig.OllamaURL = *fileCfg.OllamaURL
			}
			if fileCfg.Model != nil {
				mergedConfig.Model = *fileCfg.Model
			}
			if fileCfg.MaxTokens != nil {
				mergedConfig.MaxTokens = *fileCfg.MaxTokens
			}
			if fileCfg.Stop != nil {
				mergedConfig.Stop = *fileCfg.Stop
			}
			if fileCfg.Temperature != nil {
				mergedConfig.Temperature = *fileCfg.Temperature
			}
			if fileCfg.UseAst != nil {
				mergedConfig.UseAst = *fileCfg.UseAst
			}
			if fileCfg.UseFim != nil {
				mergedConfig.UseFim = *fileCfg.UseFim
			}
			if fileCfg.MaxPreambleLen != nil {
				mergedConfig.MaxPreambleLen = *fileCfg.MaxPreambleLen
			}
			if fileCfg.MaxSnippetLen != nil {
				mergedConfig.MaxSnippetLen = *fileCfg.MaxSnippetLen
			}

			// Update the completer's config (this handles locking).
			if err := completer.UpdateConfig(mergedConfig); err != nil {
				log.Printf("Error applying updated configuration: %v", err)
			} else {
				log.Println("Successfully applied updated configuration from client.")
			}
		} else {
			log.Println("Warning: Cannot apply configuration changes, completer not initialized.")
		}

	case "$/cancelRequest":
		log.Println("Handling '$/cancelRequest' notification...")
		var params CancelParams
		if err := json.Unmarshal(notif.Params, &params); err != nil {
			log.Printf("Error decoding cancelRequest params: %v", err)
			return
		}
		log.Printf("Received cancellation request for ID: %v", params.ID)
		// TODO (Cycle 1): Implement cancellation logic using contexts associated with request IDs.

	default:
		// Ignore other notifications.
		log.Printf("Ignoring unhandled notification method: %s", notif.Method)
	}
}

// ----------------------------------------------------------------------------
