package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync" // Added in Cycle 2
	// Import the core deepcomplete package to use its exported functions/types
	// "github.com/shehackedyou/deepcomplete" // Use your actual module path
)

// JSON-RPC message structures (basic)
type RequestMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id,omitempty"` // Can be string, int, or null
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type ResponseMessage struct {
	JSONRPC string       `json:"jsonrpc"`
	ID      interface{}  `json:"id,omitempty"`
	Result  interface{}  `json:"result,omitempty"`
	Error   *ErrorObject `json:"error,omitempty"`
}

type NotificationMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type ErrorObject struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// LSP Specific Structures (Simplified)
// --- Document URI ---
type DocumentURI string

// --- Initialize ---
type InitializeParams struct {
	ProcessID    int                `json:"processId,omitempty"`
	RootURI      DocumentURI        `json:"rootUri,omitempty"`
	ClientInfo   *ClientInfo        `json:"clientInfo,omitempty"`
	Capabilities ClientCapabilities `json:"capabilities"`
	// ... other fields
}

type ClientInfo struct {
	Name    string `json:"name,omitempty"`
	Version string `json:"version,omitempty"`
}

// Define a minimal set of capabilities for now
type ClientCapabilities struct {
	// We'll add more specific capabilities later
}

type InitializeResult struct {
	Capabilities ServerCapabilities `json:"capabilities"`
	ServerInfo   *ServerInfo        `json:"serverInfo,omitempty"`
}

type ServerCapabilities struct {
	TextDocumentSync   *TextDocumentSyncOptions `json:"textDocumentSync,omitempty"`
	CompletionProvider *CompletionOptions       `json:"completionProvider,omitempty"`
	// Add other capabilities as needed
}

type TextDocumentSyncOptions struct {
	OpenClose bool                 `json:"openClose,omitempty"` // Supports didOpen/didClose notifications
	Change    TextDocumentSyncKind `json:"change,omitempty"`    // Specifies how changes are synced (None, Full, Incremental)
}

type TextDocumentSyncKind int

const (
	// TextDocumentSyncKindNone Documents should not be synced at all.
	TextDocumentSyncKindNone TextDocumentSyncKind = 0
	// TextDocumentSyncKindFull Documents are synced by sending the full content of the document.
	TextDocumentSyncKindFull TextDocumentSyncKind = 1
	// TextDocumentSyncKindIncremental Documents are synced by sending incremental changes.
	TextDocumentSyncKindIncremental TextDocumentSyncKind = 2
)

type CompletionOptions struct {
	// We might add trigger characters, resolveProvider etc. later
}

type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
}

// --- Text Document Sync ---
type DidOpenTextDocumentParams struct {
	TextDocument TextDocumentItem `json:"textDocument"`
}

type DidCloseTextDocumentParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

type TextDocumentItem struct {
	URI        DocumentURI `json:"uri"`
	LanguageID string      `json:"languageId"`
	Version    int         `json:"version"`
	Text       string      `json:"text"`
}

type TextDocumentIdentifier struct {
	URI DocumentURI `json:"uri"`
}

// --- Document Store (Added in Cycle 2) ---
var (
	documentStore = make(map[DocumentURI][]byte)
	docStoreMutex sync.RWMutex
)

// ============================================================================
// Main Server Logic
// ============================================================================

func main() {
	// Set up logging
	logFile, err := os.OpenFile("deepcomplete-lsp.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0660)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}
	defer logFile.Close()
	log.SetOutput(logFile)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	log.Println("DeepComplete LSP server starting...")

	// TODO: Initialize DeepCompleter core service
	// completer, err := deepcomplete.NewDeepCompleter()
	// if err != nil { ... handle error ... }
	// defer completer.Close()

	reader := bufio.NewReader(os.Stdin)
	writer := os.Stdout

	for {
		// Read JSON-RPC message
		content, err := readMessage(reader)
		if err != nil {
			if err == io.EOF {
				log.Println("Client closed connection (EOF).")
				break // Exit loop on EOF
			}
			log.Printf("Error reading message: %v", err)
			continue // Continue reading? Or should we exit? Decide error strategy.
		}

		log.Printf("Received message: %s", string(content))

		// Decode JSON-RPC message
		var baseMessage RequestMessage // Use RequestMessage first to check for ID
		if err := json.Unmarshal(content, &baseMessage); err != nil {
			log.Printf("Error decoding base JSON message: %v", err)
			// TODO: Send parse error response if ID exists?
			continue
		}

		ctx := context.Background() // Create a new context for each request/notification

		// Handle Request or Notification
		if baseMessage.ID != nil {
			// It's a request
			handleRequest(ctx, baseMessage, writer)
		} else {
			// It's a notification
			handleNotification(ctx, baseMessage)
		}
	}

	log.Println("DeepComplete LSP server shutting down.")
}

// readMessage reads a single JSON-RPC message based on Content-Length header.
func readMessage(reader *bufio.Reader) ([]byte, error) {
	var contentLength int
	// Read headers
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("failed reading header line: %w", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			// End of headers
			break
		}
		if _, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength); err == nil {
			// Found Content-Length
		}
		// Ignore other headers for now
	}

	if contentLength <= 0 {
		return nil, fmt.Errorf("missing or invalid Content-Length header")
	}

	// Read body
	body := make([]byte, contentLength)
	n, err := io.ReadFull(reader, body)
	if err != nil {
		return nil, fmt.Errorf("failed reading body (read %d/%d bytes): %w", n, contentLength, err)
	}

	return body, nil
}

// writeMessage encodes and writes a JSON-RPC message with headers.
func writeMessage(writer io.Writer, message interface{}) error {
	content, err := json.Marshal(message)
	if err != nil {
		// Log internally, don't send marshalling errors back to client easily
		log.Printf("Error marshalling response: %v (Message: %+v)", err, message)
		// Optionally send a generic error response if possible?
		return fmt.Errorf("error marshalling response: %w", err)
	}

	log.Printf("Sending message: %s", string(content))

	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(content))
	_, err = writer.Write([]byte(header))
	if err != nil {
		return fmt.Errorf("error writing header: %w", err)
	}
	_, err = writer.Write(content)
	if err != nil {
		return fmt.Errorf("error writing content: %w", err)
	}
	// Flush if writer is buffered? Depends on os.Stdout behavior. Usually not needed.
	return nil
}

// handleRequest routes incoming requests based on method.
func handleRequest(ctx context.Context, req RequestMessage, writer io.Writer) {
	var response ResponseMessage
	response.JSONRPC = "2.0"
	response.ID = req.ID

	switch req.Method {
	case "initialize":
		log.Println("Handling 'initialize' request...")
		var params InitializeParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			response.Error = &ErrorObject{Code: -32602, Message: "Invalid params for initialize"}
		} else {
			// Basic capabilities for now
			response.Result = InitializeResult{
				Capabilities: ServerCapabilities{
					TextDocumentSync: &TextDocumentSyncOptions{
						OpenClose: true,                     // We handle didOpen/didClose (Cycle 2)
						Change:    TextDocumentSyncKindFull, // Start with Full sync
					},
					// CompletionProvider: &CompletionOptions{}, // Advertise later
				},
				ServerInfo: &ServerInfo{
					Name:    "DeepComplete LSP",
					Version: "0.0.1", // TODO: Get version properly
				},
			}
			log.Printf("Client Info: Name=%s, Version=%s", params.ClientInfo.Name, params.ClientInfo.Version)
		}

	case "shutdown":
		log.Println("Handling 'shutdown' request...")
		// Perform cleanup if needed before exit notification
		response.Result = nil // Success is null result

	// TODO: Add other request handlers (e.g., textDocument/completion)

	default:
		log.Printf("Received unhandled request method: %s", req.Method)
		response.Error = &ErrorObject{Code: -32601, Message: fmt.Sprintf("Method not found: %s", req.Method)}
	}

	if err := writeMessage(writer, response); err != nil {
		log.Printf("Error sending response for ID %v: %v", req.ID, err)
	}
}

// handleNotification routes incoming notifications based on method.
func handleNotification(ctx context.Context, notif NotificationMessage) {
	switch notif.Method {
	case "initialized":
		// Client confirms initialization. Maybe start background tasks?
		log.Println("Received 'initialized' notification from client.")

	case "exit":
		log.Println("Handling 'exit' notification. Server exiting.")
		// Specification requires server process to exit cleanly.
		os.Exit(0) // TODO: Check if shutdown was received first? Error code?

	case "textDocument/didOpen": // Added in Cycle 2
		log.Println("Handling 'textDocument/didOpen' notification...")
		var params DidOpenTextDocumentParams
		if err := json.Unmarshal(notif.Params, &params); err != nil {
			log.Printf("Error decoding didOpen params: %v", err)
			return
		}
		docStoreMutex.Lock()
		documentStore[params.TextDocument.URI] = []byte(params.TextDocument.Text)
		docStoreMutex.Unlock()
		log.Printf("Opened and stored document: %s (Version: %d, Length: %d)",
			params.TextDocument.URI, params.TextDocument.Version, len(params.TextDocument.Text))

	case "textDocument/didClose": // Added in Cycle 2
		log.Println("Handling 'textDocument/didClose' notification...")
		var params DidCloseTextDocumentParams
		if err := json.Unmarshal(notif.Params, &params); err != nil {
			log.Printf("Error decoding didClose params: %v", err)
			return
		}
		docStoreMutex.Lock()
		delete(documentStore, params.TextDocument.URI)
		docStoreMutex.Unlock()
		log.Printf("Closed and removed document: %s", params.TextDocument.URI)

	// TODO: Add other notification handlers (e.g., textDocument/didChange, didSave, $/cancelRequest)

	default:
		log.Printf("Received unhandled notification method: %s", notif.Method)
	}
}

// --- Functions moved to deepcomplete.go ---
// func lspPositionToBytePosition(...) { ... }
// func utf16OffsetToBytes(...) { ... }
// ---
