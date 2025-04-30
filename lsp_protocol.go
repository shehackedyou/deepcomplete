// deepcomplete/lsp_protocol.go
// Contains LSP specific data structures and utility functions
// moved from cmd/deepcomplete-lsp/main.go
package deepcomplete

import (
	"encoding/json"
	"errors"   // Needed for errors in helpers
	"fmt"      // Needed for errors in helpers
	"go/ast"   // Needed for nodeRangeToLSPRange
	"go/token" // Needed for tokenPosToLSPLocation, nodeRangeToLSPRange
	"go/types" // Needed for mapTypeToCompletionKind
	"log/slog" // Needed for helpers

	// Needed for tokenPosToLSPLocation
	// Needed for formatObjectForHover (if moved back here)
	"unicode/utf8" // Needed for byteOffsetToLSPPosition, bytesToUTF16Offset
)

// ============================================================================
// LSP Specific Structures
// ============================================================================

// DocumentURI represents the URI for a text document.
type DocumentURI string

// LSPPosition represents a 0-based line/character offset (LSP standard: UTF-16).
type LSPPosition struct {
	Line      uint32 `json:"line"`      // 0-based
	Character uint32 `json:"character"` // 0-based, UTF-16 offset
}

// LSPRange represents a range in a text document using LSP Positions (UTF-16).
type LSPRange struct {
	Start LSPPosition `json:"start"`
	End   LSPPosition `json:"end"`
}

// Location represents a location inside a resource, such as a line inside a text file.
type Location struct {
	URI   DocumentURI `json:"uri"`
	Range LSPRange    `json:"range"` // Uses LSP Range
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
	Hover      *HoverClientCapabilities      `json:"hover,omitempty"`
	Definition *DefinitionClientCapabilities `json:"definition,omitempty"`
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
type HoverClientCapabilities struct {
	ContentFormat []MarkupKind `json:"contentFormat,omitempty"` // e.g., ["markdown", "plaintext"]
}

// DefinitionClientCapabilities client capabilities for definition.
type DefinitionClientCapabilities struct {
	LinkSupport bool `json:"linkSupport,omitempty"` // Example: If client supports LocationLink
}

// InitializeResult result of the initialize request.
type InitializeResult struct {
	Capabilities ServerCapabilities `json:"capabilities"`
	ServerInfo   *ServerInfo        `json:"serverInfo,omitempty"`
}

// ServerCapabilities capabilities provided by the server.
type ServerCapabilities struct {
	TextDocumentSync   *TextDocumentSyncOptions `json:"textDocumentSync,omitempty"`
	CompletionProvider *CompletionOptions       `json:"completionProvider,omitempty"`
	HoverProvider      bool                     `json:"hoverProvider,omitempty"`
	DefinitionProvider bool                     `json:"definitionProvider,omitempty"`
	// DiagnosticProvider *DiagnosticOptions `json:"diagnosticProvider,omitempty"` // Optional
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
	Version string `json:"version,omitempty"`
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
	Position     LSPPosition            `json:"position"` // LSP Position (UTF-16)
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
	// TODO: Add preselect, sortText, filterText etc. later if needed
}

// CompletionItemKind defines the kind of completion item.
// See https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#completionItemKind
type CompletionItemKind int // Standard LSP kinds

const (
	CompletionItemKindText          CompletionItemKind = 1
	CompletionItemKindMethod        CompletionItemKind = 2
	CompletionItemKindFunction      CompletionItemKind = 3
	CompletionItemKindConstructor   CompletionItemKind = 4
	CompletionItemKindField         CompletionItemKind = 5
	CompletionItemKindVariable      CompletionItemKind = 6
	CompletionItemKindClass         CompletionItemKind = 7
	CompletionItemKindInterface     CompletionItemKind = 8
	CompletionItemKindModule        CompletionItemKind = 9
	CompletionItemKindProperty      CompletionItemKind = 10
	CompletionItemKindUnit          CompletionItemKind = 11
	CompletionItemKindValue         CompletionItemKind = 12
	CompletionItemKindEnum          CompletionItemKind = 13
	CompletionItemKindKeyword       CompletionItemKind = 14
	CompletionItemKindSnippet       CompletionItemKind = 15 // ** MODIFIED: Cycle 1 - Now default for model suggestions **
	CompletionItemKindColor         CompletionItemKind = 16
	CompletionItemKindFile          CompletionItemKind = 17
	CompletionItemKindReference     CompletionItemKind = 18
	CompletionItemKindFolder        CompletionItemKind = 19
	CompletionItemKindEnumMember    CompletionItemKind = 20
	CompletionItemKindConstant      CompletionItemKind = 21
	CompletionItemKindStruct        CompletionItemKind = 22
	CompletionItemKindEvent         CompletionItemKind = 23
	CompletionItemKindOperator      CompletionItemKind = 24
	CompletionItemKindTypeParameter CompletionItemKind = 25
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

// HoverParams parameters for textDocument/hover.
type HoverParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     LSPPosition            `json:"position"` // LSP Position (UTF-16)
}

// HoverResult result for textDocument/hover.
type HoverResult struct {
	Contents MarkupContent `json:"contents"`
	Range    *LSPRange     `json:"range,omitempty"` // Optional: range of the hovered symbol (LSP Range)
}

// MarkupContent represents structured content for hover/documentation.
type MarkupContent struct {
	Kind  MarkupKind `json:"kind"` // e.g., "markdown" or "plaintext"
	Value string     `json:"value"`
}

// MarkupKind defines the kind of markup content.
type MarkupKind string

const (
	MarkupKindPlainText MarkupKind = "plaintext"
	MarkupKindMarkdown  MarkupKind = "markdown"
)

// DefinitionParams parameters for textDocument/definition.
type DefinitionParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     LSPPosition            `json:"position"` // LSP Position (UTF-16)
}

// DefinitionResult can be Location, []Location, or LocationLink[]
type DefinitionResult = []Location // Using []Location for simplicity

// ShowMessageParams parameters for window/showMessage notification.
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

// LspDiagnosticSeverity defines the severity level of a diagnostic (LSP Standard).
type LspDiagnosticSeverity int

const (
	LspSeverityError   LspDiagnosticSeverity = 1
	LspSeverityWarning LspDiagnosticSeverity = 2
	LspSeverityInfo    LspDiagnosticSeverity = 3
	LspSeverityHint    LspDiagnosticSeverity = 4
)

// LspDiagnostic represents a diagnostic (LSP Standard).
type LspDiagnostic struct {
	Range    LSPRange              `json:"range"`            // The range (LSP UTF-16) at which the message applies.
	Severity LspDiagnosticSeverity `json:"severity"`         // The diagnostic's severity.
	Code     any                   `json:"code,omitempty"`   // The diagnostic's code, which might be a number or string.
	Source   string                `json:"source,omitempty"` // A human-readable string describing the source of this diagnostic, e.g. 'go' or 'deepcomplete'.
	Message  string                `json:"message"`          // The diagnostic's message.
	// RelatedInformation []DiagnosticRelatedInformation `json:"relatedInformation,omitempty"` // Optional related locations.
	// Tags []DiagnosticTag `json:"tags,omitempty"` // Optional tags like Unnecessary or Deprecated.
}

// PublishDiagnosticsParams parameters for textDocument/publishDiagnostics notification.
type PublishDiagnosticsParams struct {
	URI         DocumentURI     `json:"uri"`
	Version     *int            `json:"version,omitempty"` // Optional: The version number of the document the diagnostics are published for.
	Diagnostics []LspDiagnostic `json:"diagnostics"`       // An array of diagnostic items (LSP Diagnostics).
}

// ============================================================================
// JSON-RPC Structures
// ============================================================================

// JSON-RPC Standard Error Codes
const (
	JsonRpcParseError           int = -32700
	JsonRpcInvalidRequest       int = -32600
	JsonRpcMethodNotFound       int = -32601
	JsonRpcInvalidParams        int = -32602
	JsonRpcInternalError        int = -32603
	JsonRpcRequestCancelled     int = -32800
	JsonRpcServerNotInitialized int = -32002
	JsonRpcServerBusy           int = -32000
	JsonRpcRequestFailed        int = -32803
)

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
// LSP Utility Functions (Moved from cmd/deepcomplete-lsp/main.go and utils)
// ============================================================================

// mapTypeToCompletionKind maps a Go types.Object to an LSP CompletionItemKind.
// (Moved from lsp_server.go)
// ** MODIFIED: Cycle 1 - Default to Snippet for model suggestions **
func mapTypeToCompletionKind(obj types.Object) CompletionItemKind {
	if obj == nil {
		// Default to Snippet kind for completions generated without specific type info (e.g., LLM suggestions)
		return CompletionItemKindSnippet
	}

	switch o := obj.(type) {
	case *types.Func:
		sig, ok := o.Type().(*types.Signature)
		if ok && sig.Recv() != nil {
			return CompletionItemKindMethod
		}
		return CompletionItemKindFunction
	case *types.Var:
		if o.IsField() {
			return CompletionItemKindField
		}
		return CompletionItemKindVariable
	case *types.Const:
		return CompletionItemKindConstant
	case *types.TypeName:
		switch o.Type().Underlying().(type) {
		case *types.Struct:
			return CompletionItemKindStruct
		case *types.Interface:
			return CompletionItemKindInterface
		// Add cases for *types.Basic (like int, string), *types.Map, *types.Slice, etc. if needed
		default:
			return CompletionItemKindClass // General fallback for types
		}
	case *types.PkgName:
		return CompletionItemKindModule
	case *types.Builtin:
		// Could potentially map specific builtins (e.g., append, make) to Function?
		return CompletionItemKindFunction // Treat builtins like functions
	case *types.Nil:
		return CompletionItemKindValue
	default:
		// Fallback for other known types.Object types
		return CompletionItemKindText
	}
}

// tokenPosToLSPLocation converts a token.Pos to an LSP Location.
// Requires the token.File containing the position and the file content.
// (Moved from lsp_server.go)
func tokenPosToLSPLocation(file *token.File, pos token.Pos, content []byte, logger *slog.Logger) (*Location, error) {
	if file == nil {
		return nil, errors.New("cannot convert position: token.File is nil")
	}
	if !pos.IsValid() {
		return nil, errors.New("cannot convert position: token.Pos is invalid")
	}

	// Get the 0-based offset within the file
	offset := file.Offset(pos)
	if offset < 0 || offset > file.Size() {
		return nil, fmt.Errorf("invalid offset %d calculated from pos %d in file %s (size %d)", offset, pos, file.Name(), file.Size())
	}

	// Convert 0-based byte offset to 0-based LSP line/char (UTF-16)
	lspLine, lspChar, convErr := byteOffsetToLSPPosition(content, offset, logger) // Use function defined below
	if convErr != nil {
		return nil, fmt.Errorf("failed converting byte offset %d to LSP position: %w", offset, convErr)
	}

	// Construct file URI
	fileURIStr, uriErr := PathToURI(file.Name()) // Use utility function from deepcomplete_utils.go
	if uriErr != nil {
		logger.Warn("Failed to convert definition file path to URI", "path", file.Name(), "error", uriErr)
		return nil, fmt.Errorf("failed to create URI for definition file %s: %w", file.Name(), uriErr)
	}
	lspFileURI := DocumentURI(fileURIStr)

	// Create LSP Position and Range (range spans just the single point for now)
	lspPosition := LSPPosition{Line: lspLine, Character: lspChar}
	lspRange := LSPRange{Start: lspPosition, End: lspPosition}

	return &Location{
		URI:   lspFileURI,
		Range: lspRange,
	}, nil
}

// byteOffsetToLSPPosition converts a 0-based byte offset to 0-based LSP line/char (UTF-16).
// (Moved from lsp_server.go)
func byteOffsetToLSPPosition(content []byte, targetByteOffset int, logger *slog.Logger) (line, char uint32, err error) {
	if content == nil {
		return 0, 0, errors.New("content is nil")
	}
	if targetByteOffset < 0 {
		return 0, 0, fmt.Errorf("invalid targetByteOffset: %d", targetByteOffset)
	}
	if targetByteOffset > len(content) {
		targetByteOffset = len(content)
		logger.Debug("targetByteOffset exceeds content length, clamping to EOF", "offset", targetByteOffset, "content_len", len(content))
	}

	currentLine := uint32(0)
	currentByteOffset := 0
	currentLineStartByteOffset := 0

	for currentByteOffset < targetByteOffset {
		r, size := utf8.DecodeRune(content[currentByteOffset:])
		if r == utf8.RuneError && size <= 1 {
			return 0, 0, fmt.Errorf("invalid UTF-8 sequence at byte offset %d", currentByteOffset)
		}
		if r == '\n' {
			currentLine++
			currentLineStartByteOffset = currentByteOffset + size
		}
		currentByteOffset += size
	}

	lineContentBytes := content[currentLineStartByteOffset:targetByteOffset]
	utf16CharOffset, convErr := bytesToUTF16Offset(lineContentBytes, logger) // Use function defined below
	if convErr != nil {
		logger.Error("Error converting line bytes to UTF16 offset", "error", convErr, "line", currentLine)
		utf16CharOffset = len(lineContentBytes) // Fallback
	}

	return currentLine, uint32(utf16CharOffset), nil
}

// bytesToUTF16Offset calculates the number of UTF-16 code units for a byte slice.
// (Moved from lsp_server.go)
func bytesToUTF16Offset(bytes []byte, logger *slog.Logger) (int, error) {
	utf16Offset := 0
	byteOffset := 0
	for byteOffset < len(bytes) {
		r, size := utf8.DecodeRune(bytes[byteOffset:])
		if r == utf8.RuneError && size <= 1 {
			return utf16Offset, fmt.Errorf("%w at byte offset %d within slice", ErrInvalidUTF8, byteOffset) // Use exported error
		}
		if r > 0xFFFF {
			utf16Offset += 2 // Surrogate pair
		} else {
			utf16Offset += 1
		}
		byteOffset += size
	}
	return utf16Offset, nil
}

// nodeRangeToLSPRange converts an AST node's position range to an LSP Range.
// (Moved from lsp_server.go)
func nodeRangeToLSPRange(fset *token.FileSet, node ast.Node, content []byte, logger *slog.Logger) (*LSPRange, error) {
	if fset == nil || node == nil {
		return nil, errors.New("fset or node is nil")
	}
	startTokenPos := node.Pos()
	endTokenPos := node.End()

	if !startTokenPos.IsValid() || !endTokenPos.IsValid() {
		return nil, errors.New("node position is invalid")
	}

	file := fset.File(startTokenPos)
	if file == nil {
		return nil, errors.New("could not get token.File for node")
	}

	startOffset := file.Offset(startTokenPos)
	endOffset := file.Offset(endTokenPos)

	if startOffset < 0 || endOffset < 0 || startOffset > len(content) || endOffset > len(content) || endOffset < startOffset {
		return nil, fmt.Errorf("invalid byte offsets calculated: start=%d, end=%d, content_len=%d", startOffset, endOffset, len(content))
	}

	startLine, startChar, startErr := byteOffsetToLSPPosition(content, startOffset, logger) // Use function defined above
	endLine, endChar, endErr := byteOffsetToLSPPosition(content, endOffset, logger)         // Use function defined above

	if startErr != nil || endErr != nil {
		return nil, fmt.Errorf("failed converting offsets to LSP positions: startErr=%v, endErr=%v", startErr, endErr)
	}

	return &LSPRange{
		Start: LSPPosition{Line: startLine, Character: startChar},
		End:   LSPPosition{Line: endLine, Character: endChar},
	}, nil
}

// NOTE: formatObjectForHover remains in deepcomplete_helpers.go as it depends
// heavily on AstContextInfo which is defined there.
