// deepcomplete/lsp_protocol.go
// Contains LSP specific data structures and utility functions.
package deepcomplete

import (
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"   // Needed for nodeRangeToLSPRange
	"go/token" // Needed for tokenPosToLSPLocation, nodeRangeToLSPRange
	"go/types" // Needed for mapTypeToCompletionKind
	"log/slog"
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
}

// TextDocumentClientCapabilities text document specific client capabilities.
type TextDocumentClientCapabilities struct {
	Completion *CompletionClientCapabilities `json:"completion,omitempty"`
	Hover      *HoverClientCapabilities      `json:"hover,omitempty"`
	Definition *DefinitionClientCapabilities `json:"definition,omitempty"`
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
	LinkSupport bool `json:"linkSupport,omitempty"`
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
	Settings json.RawMessage `json:"settings"` // Can be anything
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
	Documentation    string             `json:"documentation,omitempty"`    // Documentation string
	InsertTextFormat InsertTextFormat   `json:"insertTextFormat,omitempty"` // PlainText or Snippet
	InsertText       string             `json:"insertText,omitempty"`       // Text to insert
}

// CompletionItemKind defines the kind of completion item (LSP standard).
type CompletionItemKind int

const (
	CompletionItemKindText          CompletionItemKind = 1
	CompletionItemKindMethod        CompletionItemKind = 2
	CompletionItemKindFunction      CompletionItemKind = 3
	CompletionItemKindConstructor   CompletionItemKind = 4
	CompletionItemKindField         CompletionItemKind = 5
	CompletionItemKindVariable      CompletionItemKind = 6
	CompletionItemKindClass         CompletionItemKind = 7 // Often used for Type specs
	CompletionItemKindInterface     CompletionItemKind = 8
	CompletionItemKindModule        CompletionItemKind = 9 // Often used for Packages
	CompletionItemKindProperty      CompletionItemKind = 10
	CompletionItemKindUnit          CompletionItemKind = 11
	CompletionItemKindValue         CompletionItemKind = 12
	CompletionItemKindEnum          CompletionItemKind = 13
	CompletionItemKindKeyword       CompletionItemKind = 14
	CompletionItemKindSnippet       CompletionItemKind = 15 // Default for model suggestions
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
	Range    *LSPRange     `json:"range,omitempty"` // Optional: range of the hovered symbol
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
	Code     any                   `json:"code,omitempty"`   // The diagnostic's code (number or string).
	Source   string                `json:"source,omitempty"` // e.g. 'go' or 'deepcomplete'.
	Message  string                `json:"message"`          // The diagnostic's message.
}

// PublishDiagnosticsParams parameters for textDocument/publishDiagnostics notification.
type PublishDiagnosticsParams struct {
	URI         DocumentURI     `json:"uri"`
	Version     *int            `json:"version,omitempty"` // Optional: Document version.
	Diagnostics []LspDiagnostic `json:"diagnostics"`       // Array of diagnostic items.
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
// LSP Utility Functions
// ============================================================================

// mapTypeToCompletionKind maps a Go types.Object to an LSP CompletionItemKind.
// Provides more specific kinds than the previous version.
func mapTypeToCompletionKind(obj types.Object) CompletionItemKind {
	if obj == nil {
		return CompletionItemKindSnippet // Default for LLM suggestions without type info
	}

	switch o := obj.(type) {
	case *types.Func:
		sig, ok := o.Type().(*types.Signature)
		if ok && sig.Recv() != nil {
			return CompletionItemKindMethod
		}
		// Could potentially check if it's a constructor-like function
		return CompletionItemKindFunction
	case *types.Var:
		if o.IsField() {
			return CompletionItemKindField
		}
		// Check if it's a package-level variable
		if o.Parent() != nil && o.Parent() == o.Pkg().Scope() {
			// Could potentially treat as Constant if it looks like one, but Variable is safer
			return CompletionItemKindVariable
		}
		return CompletionItemKindVariable
	case *types.Const:
		// Check underlying type if possible
		if t := o.Type(); t != nil {
			if basic, ok := t.Underlying().(*types.Basic); ok {
				// Check if it's boolean or numeric-like constant
				if basic.Info()&(types.IsBoolean|types.IsNumeric) != 0 {
					return CompletionItemKindConstant
				}
				// Could check for string constants etc.
			}
		}
		return CompletionItemKindConstant // Default for const
	case *types.TypeName:
		// Get the underlying type to determine the specific kind
		underlying := o.Type().Underlying()
		if underlying == nil {
			return CompletionItemKindClass // Fallback for type names
		}
		switch underlying.(type) {
		case *types.Struct:
			return CompletionItemKindStruct
		case *types.Interface:
			return CompletionItemKindInterface
		case *types.Basic:
			// Could differentiate basic types (int, string) further if needed
			return CompletionItemKindKeyword // Often used for basic type keywords
		case *types.Signature: // Type alias for a function type
			return CompletionItemKindFunction
		case *types.Map:
			return CompletionItemKindClass // Or Struct/Keyword depending on preference
		case *types.Slice, *types.Array:
			return CompletionItemKindClass // Or Struct/Keyword
		case *types.Pointer:
			// Look at the element type of the pointer
			if elem := underlying.(*types.Pointer).Elem(); elem != nil {
				// Recursively map the element type (avoid infinite recursion for self-referential types)
				// For simplicity, just return Class for pointers for now.
				return CompletionItemKindClass
			}
			return CompletionItemKindClass
		default:
			return CompletionItemKindClass // General fallback for other type kinds
		}
	case *types.PkgName:
		return CompletionItemKindModule
	case *types.Builtin:
		return CompletionItemKindFunction // Built-in functions like make, append
	case *types.Nil:
		return CompletionItemKindValue // `nil` is a value
	case *types.Label:
		return CompletionItemKindReference // Labels are like references
	default:
		return CompletionItemKindText // Safest generic fallback
	}
}

// tokenPosToLSPLocation converts a token.Pos to an LSP Location.
// Requires the token.File containing the position and the file content.
func tokenPosToLSPLocation(file *token.File, pos token.Pos, content []byte, logger *slog.Logger) (*Location, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if file == nil {
		return nil, errors.New("cannot convert position: token.File is nil")
	}
	if !pos.IsValid() {
		return nil, errors.New("cannot convert position: token.Pos is invalid")
	}
	if content == nil {
		return nil, errors.New("cannot convert position: file content is nil")
	}

	offset := file.Offset(pos)
	if offset < 0 || offset > file.Size() || offset > len(content) {
		return nil, fmt.Errorf("invalid offset %d calculated from pos %d in file %s (size %d, content len %d)", offset, pos, file.Name(), file.Size(), len(content))
	}

	lspLine, lspChar, convErr := byteOffsetToLSPPosition(content, offset, logger) // Pass logger
	if convErr != nil {
		return nil, fmt.Errorf("failed converting byte offset %d to LSP position: %w", offset, convErr)
	}

	fileURIStr, uriErr := PathToURI(file.Name())
	if uriErr != nil {
		logger.Warn("Failed to convert definition file path to URI", "path", file.Name(), "error", uriErr)
		return nil, fmt.Errorf("failed to create URI for definition file %s: %w", file.Name(), uriErr)
	}
	lspFileURI := DocumentURI(fileURIStr)

	lspPosition := LSPPosition{Line: lspLine, Character: lspChar}
	lspRange := LSPRange{Start: lspPosition, End: lspPosition} // Point range

	return &Location{
		URI:   lspFileURI,
		Range: lspRange,
	}, nil
}

// nodeRangeToLSPRange converts an AST node's position range to an LSP Range.
func nodeRangeToLSPRange(fset *token.FileSet, node ast.Node, content []byte, logger *slog.Logger) (*LSPRange, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if fset == nil || node == nil {
		return nil, errors.New("fset or node is nil")
	}
	startTokenPos := node.Pos()
	endTokenPos := node.End()

	if !startTokenPos.IsValid() || !endTokenPos.IsValid() {
		return nil, fmt.Errorf("node position is invalid (start: %v, end: %v)", startTokenPos.IsValid(), endTokenPos.IsValid())
	}

	file := fset.File(startTokenPos)
	if file == nil {
		return nil, fmt.Errorf("could not get token.File for node starting at pos %d", startTokenPos)
	}
	if fset.File(endTokenPos) != file {
		return nil, fmt.Errorf("node spans multiple files (start: %s, end: %s)", file.Name(), fset.File(endTokenPos).Name())
	}
	if content == nil {
		return nil, errors.New("cannot convert node range: file content is nil")
	}

	startOffset := file.Offset(startTokenPos)
	endOffset := file.Offset(endTokenPos)
	fileSize := file.Size()
	contentLen := len(content)

	if startOffset < 0 || endOffset < 0 || startOffset > fileSize || endOffset > fileSize || startOffset > contentLen || endOffset > contentLen || endOffset < startOffset {
		return nil, fmt.Errorf("invalid byte offsets calculated: start=%d, end=%d, file_size=%d, content_len=%d", startOffset, endOffset, fileSize, contentLen)
	}

	startLine, startChar, startErr := byteOffsetToLSPPosition(content, startOffset, logger) // Pass logger
	endLine, endChar, endErr := byteOffsetToLSPPosition(content, endOffset, logger)         // Pass logger

	if startErr != nil || endErr != nil {
		return nil, fmt.Errorf("failed converting offsets to LSP positions: startErr=%v, endErr=%v", startErr, endErr)
	}

	return &LSPRange{
		Start: LSPPosition{Line: startLine, Character: startChar},
		End:   LSPPosition{Line: endLine, Character: endChar},
	}, nil
}
