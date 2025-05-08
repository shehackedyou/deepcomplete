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
	Trace                 string             `json:"trace,omitempty"` // off, messages, verbose
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
	Window       *WindowClientCapabilities       `json:"window,omitempty"`
	Experimental any                             `json:"experimental,omitempty"`
}

// WorkspaceClientCapabilities workspace specific client capabilities.
type WorkspaceClientCapabilities struct {
	ApplyEdit              bool                                `json:"applyEdit,omitempty"`
	DidChangeConfiguration *DidChangeConfigurationCapabilities `json:"didChangeConfiguration,omitempty"`
	// Add other workspace capabilities like workspaceFolders, symbol, executeCommand etc. if needed
}

// DidChangeConfigurationCapabilities capabilities for workspace/didChangeConfiguration.
type DidChangeConfigurationCapabilities struct {
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// TextDocumentClientCapabilities text document specific client capabilities.
type TextDocumentClientCapabilities struct {
	Completion    *CompletionClientCapabilities    `json:"completion,omitempty"`
	Hover         *HoverClientCapabilities         `json:"hover,omitempty"`
	Definition    *DefinitionClientCapabilities    `json:"definition,omitempty"`
	SignatureHelp *SignatureHelpClientCapabilities `json:"signatureHelp,omitempty"` // Added (Cycle N+9)
	CodeAction    *CodeActionClientCapabilities    `json:"codeAction,omitempty"`
	// Add other text document capabilities like references etc. if needed
}

// CompletionClientCapabilities client capabilities for completion.
type CompletionClientCapabilities struct {
	CompletionItem *CompletionItemClientCapabilities `json:"completionItem,omitempty"`
	// Add contextSupport etc. if needed
}

// CompletionItemClientCapabilities client capabilities specific to completion items.
type CompletionItemClientCapabilities struct {
	SnippetSupport bool `json:"snippetSupport,omitempty"`
	// Add commitCharactersSupport, documentationFormat etc. if needed
}

// HoverClientCapabilities client capabilities for hover.
type HoverClientCapabilities struct {
	ContentFormat []MarkupKind `json:"contentFormat,omitempty"` // e.g., ["markdown", "plaintext"]
}

// DefinitionClientCapabilities client capabilities for definition.
type DefinitionClientCapabilities struct {
	LinkSupport bool `json:"linkSupport,omitempty"`
}

// SignatureHelpClientCapabilities client capabilities for signature help. (Cycle N+9)
type SignatureHelpClientCapabilities struct {
	DynamicRegistration  bool                            `json:"dynamicRegistration,omitempty"`
	SignatureInformation *SignatureInformationClientCaps `json:"signatureInformation,omitempty"`
	ContextSupport       bool                            `json:"contextSupport,omitempty"`
}

// SignatureInformationClientCaps capabilities specific to SignatureInformation. (Cycle N+9)
type SignatureInformationClientCaps struct {
	DocumentationFormat    []MarkupKind                    `json:"documentationFormat,omitempty"`
	ParameterInformation   *ParameterInformationClientCaps `json:"parameterInformation,omitempty"`
	ActiveParameterSupport bool                            `json:"activeParameterSupport,omitempty"`
}

// ParameterInformationClientCaps capabilities specific to ParameterInformation. (Cycle N+9)
type ParameterInformationClientCaps struct {
	LabelOffsetSupport bool `json:"labelOffsetSupport,omitempty"`
}

// CodeActionClientCapabilities client capabilities for code actions.
type CodeActionClientCapabilities struct {
	DynamicRegistration      bool                            `json:"dynamicRegistration,omitempty"`
	CodeActionLiteralSupport *CodeActionLiteralSupportClient `json:"codeActionLiteralSupport,omitempty"`
	IsPreferredSupport       bool                            `json:"isPreferredSupport,omitempty"`
	// Add resolveSupport etc. if needed
}

// CodeActionLiteralSupportClient capabilities specific to CodeActionLiterals.
type CodeActionLiteralSupportClient struct {
	CodeActionKind CodeActionKindClientCapabilities `json:"codeActionKind"`
}

// CodeActionKindClientCapabilities defines capabilities for code action kinds.
type CodeActionKindClientCapabilities struct {
	ValueSet []CodeActionKind `json:"valueSet"` // The code action kinds the client supports
}

// WindowClientCapabilities capabilities specific to the window.
type WindowClientCapabilities struct {
	WorkDoneProgress bool `json:"workDoneProgress,omitempty"`
	// Add showMessage, showDocument etc. if needed
}

// InitializeResult result of the initialize request.
type InitializeResult struct {
	Capabilities ServerCapabilities `json:"capabilities"`
	ServerInfo   *ServerInfo        `json:"serverInfo,omitempty"`
}

// ServerCapabilities capabilities provided by the server.
type ServerCapabilities struct {
	TextDocumentSync      *TextDocumentSyncOptions `json:"textDocumentSync,omitempty"`
	CompletionProvider    *CompletionOptions       `json:"completionProvider,omitempty"`
	HoverProvider         bool                     `json:"hoverProvider,omitempty"`         // Simple boolean for now
	DefinitionProvider    bool                     `json:"definitionProvider,omitempty"`    // Simple boolean for now
	SignatureHelpProvider *SignatureHelpOptions    `json:"signatureHelpProvider,omitempty"` // Added (Cycle N+9)
	CodeActionProvider    any                      `json:"codeActionProvider,omitempty"`    // bool | CodeActionOptions
	// Add other capabilities like referencesProvider etc. if needed
}

// TextDocumentSyncOptions options for text document synchronization.
type TextDocumentSyncOptions struct {
	OpenClose bool                 `json:"openClose,omitempty"`
	Change    TextDocumentSyncKind `json:"change,omitempty"`
	// Add willSave, willSaveWaitUntil, save if needed
}

// TextDocumentSyncKind defines how text document changes are synced.
type TextDocumentSyncKind int

const (
	TextDocumentSyncKindNone        TextDocumentSyncKind = 0
	TextDocumentSyncKindFull        TextDocumentSyncKind = 1 // We only support Full sync
	TextDocumentSyncKindIncremental TextDocumentSyncKind = 2
)

// CompletionOptions server completion capabilities.
type CompletionOptions struct {
	ResolveProvider   bool     `json:"resolveProvider,omitempty"`
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`
	// Add allCommitCharacters, workDoneProgress if needed
}

// SignatureHelpOptions server capabilities for signature help. (Cycle N+9)
type SignatureHelpOptions struct {
	TriggerCharacters   []string `json:"triggerCharacters,omitempty"`
	RetriggerCharacters []string `json:"retriggerCharacters,omitempty"`
	// Add workDoneProgress if needed
}

// CodeActionOptions server capabilities for code actions.
type CodeActionOptions struct {
	CodeActionKinds []CodeActionKind `json:"codeActionKinds,omitempty"`
	ResolveProvider bool             `json:"resolveProvider,omitempty"`
	// Add workDoneProgress if needed
}

// CodeActionKind defines the kind of code action (string).
type CodeActionKind string

const (
	CodeActionKindQuickFix              CodeActionKind = "quickfix"
	CodeActionKindRefactor              CodeActionKind = "refactor"
	CodeActionKindRewrite               CodeActionKind = "refactor.rewrite"
	CodeActionKindExtract               CodeActionKind = "refactor.extract"
	CodeActionKindInline                CodeActionKind = "refactor.inline"
	CodeActionKindSource                CodeActionKind = "source"
	CodeActionKindSourceOrganizeImports CodeActionKind = "source.organizeImports"
	CodeActionKindSourceFixAll          CodeActionKind = "source.fixAll"
)

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
	ContentChanges []TextDocumentContentChangeEvent `json:"contentChanges"`
}

// VersionedTextDocumentIdentifier identifies a text document with a version number.
type VersionedTextDocumentIdentifier struct {
	TextDocumentIdentifier
	Version int `json:"version"` // Can be null if client doesn't support versioning
}

// TextDocumentContentChangeEvent an event describing a change to a text document.
type TextDocumentContentChangeEvent struct {
	// For Full sync: Range and RangeLength are omitted. Text contains the full content.
	Text string `json:"text"`
}

// DidChangeConfigurationParams parameters for workspace/didChangeConfiguration.
type DidChangeConfigurationParams struct {
	Settings json.RawMessage `json:"settings"` // Can be anything
}

// CompletionParams parameters for textDocument/completion.
type CompletionParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     LSPPosition            `json:"position"`
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
	CompletionTriggerKindInvoked              CompletionTriggerKind = 1
	CompletionTriggerKindTriggerChar          CompletionTriggerKind = 2
	CompletionTriggerKindTriggerForIncomplete CompletionTriggerKind = 3
)

// CompletionList represents a list of completion items.
type CompletionList struct {
	IsIncomplete bool             `json:"isIncomplete"`
	Items        []CompletionItem `json:"items"`
}

// CompletionItem represents a single completion suggestion.
type CompletionItem struct {
	Label            string             `json:"label"`
	Kind             CompletionItemKind `json:"kind,omitempty"`
	Detail           string             `json:"detail,omitempty"`
	Documentation    any                `json:"documentation,omitempty"` // string | MarkupContent
	InsertTextFormat InsertTextFormat   `json:"insertTextFormat,omitempty"`
	InsertText       string             `json:"insertText,omitempty"`
	// Add other fields like preselect, sortText, filterText, textEdit, additionalTextEdits, command etc.
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
	CompletionItemKindClass         CompletionItemKind = 7
	CompletionItemKindInterface     CompletionItemKind = 8
	CompletionItemKindModule        CompletionItemKind = 9
	CompletionItemKindProperty      CompletionItemKind = 10
	CompletionItemKindUnit          CompletionItemKind = 11
	CompletionItemKindValue         CompletionItemKind = 12
	CompletionItemKindEnum          CompletionItemKind = 13
	CompletionItemKindKeyword       CompletionItemKind = 14
	CompletionItemKindSnippet       CompletionItemKind = 15
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
	ID any `json:"id"` // number | string
}

// HoverParams parameters for textDocument/hover.
type HoverParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     LSPPosition            `json:"position"`
}

// HoverResult result for textDocument/hover.
type HoverResult struct {
	Contents MarkupContent `json:"contents"`
	Range    *LSPRange     `json:"range,omitempty"`
}

// MarkupContent represents structured content for hover/documentation.
type MarkupContent struct {
	Kind  MarkupKind `json:"kind"` // "plaintext" | "markdown"
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
	Position     LSPPosition            `json:"position"`
}

// DefinitionResult can be Location, []Location, or LocationLink[]
type DefinitionResult = []Location // Using []Location for simplicity

// SignatureHelpParams parameters for textDocument/signatureHelp (Cycle N+9)
type SignatureHelpParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     LSPPosition            `json:"position"`
	Context      *SignatureHelpContext  `json:"context,omitempty"`
}

// SignatureHelpContext additional information about the context of the signature help request (Cycle N+9)
type SignatureHelpContext struct {
	TriggerKind         SignatureHelpTriggerKind `json:"triggerKind"`                   // How the signature help was triggered.
	TriggerCharacter    string                   `json:"triggerCharacter,omitempty"`    // Character that triggered signature help.
	IsRetrigger         bool                     `json:"isRetrigger"`                   // `true` if signature help was already showing.
	ActiveSignatureHelp *SignatureHelp           `json:"activeSignatureHelp,omitempty"` // Active help if `isRetrigger` is `true`.
}

// SignatureHelpTriggerKind how signature help was triggered. (Cycle N+9)
type SignatureHelpTriggerKind int

const (
	SignatureHelpTriggerKindInvoked          SignatureHelpTriggerKind = 1 // Signature help was invoked manually by the user or client.
	SignatureHelpTriggerKindTriggerCharacter SignatureHelpTriggerKind = 2 // Signature help was triggered by a trigger character.
	SignatureHelpTriggerKindContentChange    SignatureHelpTriggerKind = 3 // Signature help was triggered by the cursor moving or by the document content changing.
)

// SignatureHelp represents the signature help response. (Cycle N+9)
type SignatureHelp struct {
	Signatures      []SignatureInformation `json:"signatures"`                // One or more signatures.
	ActiveSignature *uint32                `json:"activeSignature,omitempty"` // The active signature index.
	ActiveParameter *uint32                `json:"activeParameter,omitempty"` // The active parameter index.
}

// SignatureInformation represents a signature and its documentation. (Cycle N+9)
type SignatureInformation struct {
	Label           string                 `json:"label"`                     // Label of this signature. Will be shown in the UI.
	Documentation   any                    `json:"documentation,omitempty"`   // Documentation (string | MarkupContent).
	Parameters      []ParameterInformation `json:"parameters,omitempty"`      // The parameters of this signature.
	ActiveParameter *uint32                `json:"activeParameter,omitempty"` // Index of the active parameter.
}

// ParameterInformation represents a parameter of a callable signature. (Cycle N+9)
type ParameterInformation struct {
	Label         any `json:"label"`                   // Label string or [start, end] offsets.
	Documentation any `json:"documentation,omitempty"` // Documentation (string | MarkupContent).
}

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
	Range    LSPRange              `json:"range"`
	Severity LspDiagnosticSeverity `json:"severity"`
	Code     any                   `json:"code,omitempty"` // number | string
	Source   string                `json:"source,omitempty"`
	Message  string                `json:"message"`
	// Add relatedInformation, tags etc. if needed
}

// PublishDiagnosticsParams parameters for textDocument/publishDiagnostics notification.
type PublishDiagnosticsParams struct {
	URI         DocumentURI     `json:"uri"`
	Version     *int            `json:"version,omitempty"` // Use pointer to allow omitting
	Diagnostics []LspDiagnostic `json:"diagnostics"`
}

// CodeActionParams parameters for textDocument/codeAction
type CodeActionParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Range        LSPRange               `json:"range"`
	Context      CodeActionContext      `json:"context"`
}

// CodeActionContext contains context information for code action requests
type CodeActionContext struct {
	Diagnostics []LspDiagnostic       `json:"diagnostics"`
	Only        []CodeActionKind      `json:"only,omitempty"`
	TriggerKind CodeActionTriggerKind `json:"triggerKind,omitempty"`
}

// CodeActionTriggerKind defines how a code action was invoked
type CodeActionTriggerKind int

const (
	CodeActionTriggerKindInvoked   CodeActionTriggerKind = 1
	CodeActionTriggerKindAutomatic CodeActionTriggerKind = 2
)

// CodeActionResult is the result of a code action request
type CodeActionResult = []any // []Command | []CodeAction

// Command represents a command that can be executed on the client
type Command struct {
	Title     string `json:"title"`
	Command   string `json:"command"`
	Arguments []any  `json:"arguments,omitempty"`
}

// CodeAction represents a potential action offered to the user
type CodeAction struct {
	Title       string          `json:"title"`
	Kind        CodeActionKind  `json:"kind,omitempty"`
	Diagnostics []LspDiagnostic `json:"diagnostics,omitempty"`
	IsPreferred bool            `json:"isPreferred,omitempty"`
	Edit        *WorkspaceEdit  `json:"edit,omitempty"`
	Command     *Command        `json:"command,omitempty"`
	Data        any             `json:"data,omitempty"`
	// Add disabled, documentation etc. if needed
}

// WorkspaceEdit represents changes to multiple resources managed by the workspace
type WorkspaceEdit struct {
	Changes         map[DocumentURI][]TextEdit `json:"changes,omitempty"`
	DocumentChanges []TextDocumentEdit         `json:"documentChanges,omitempty"`
	// Add changeAnnotations if needed
}

// TextEdit represents a textual change in a document
type TextEdit struct {
	Range   LSPRange `json:"range"`
	NewText string   `json:"newText"`
}

// TextDocumentEdit represents edits to a specific version of a text document
type TextDocumentEdit struct {
	TextDocument OptionalVersionedTextDocumentIdentifier `json:"textDocument"`
	Edits        []TextEdit                              `json:"edits"`
}

// OptionalVersionedTextDocumentIdentifier is like VersionedTextDocumentIdentifier but version is optional
type OptionalVersionedTextDocumentIdentifier struct {
	TextDocumentIdentifier
	Version *int `json:"version"` // Version number is optional (null)
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
func mapTypeToCompletionKind(obj types.Object) CompletionItemKind {
	if obj == nil {
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
		underlying := o.Type().Underlying()
		if underlying == nil {
			return CompletionItemKindClass
		}
		switch underlying.(type) {
		case *types.Struct:
			return CompletionItemKindStruct
		case *types.Interface:
			return CompletionItemKindInterface
		case *types.Basic:
			return CompletionItemKindKeyword
		case *types.Signature:
			return CompletionItemKindFunction
		case *types.Map, *types.Slice, *types.Array, *types.Pointer:
			return CompletionItemKindClass
		default:
			return CompletionItemKindClass
		}
	case *types.PkgName:
		return CompletionItemKindModule
	case *types.Builtin:
		return CompletionItemKindFunction
	case *types.Nil:
		return CompletionItemKindValue
	case *types.Label:
		return CompletionItemKindReference
	default:
		return CompletionItemKindText
	}
}

// tokenPosToLSPLocation converts a token.Pos to an LSP Location.
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

	lspLine, lspChar, convErr := byteOffsetToLSPPosition(content, offset, logger)
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

	startLine, startChar, startErr := byteOffsetToLSPPosition(content, startOffset, logger)
	endLine, endChar, endErr := byteOffsetToLSPPosition(content, endOffset, logger)

	if startErr != nil || endErr != nil {
		return nil, fmt.Errorf("failed converting offsets to LSP positions: startErr=%v, endErr=%v", startErr, endErr)
	}

	return &LSPRange{
		Start: LSPPosition{Line: startLine, Character: startChar},
		End:   LSPPosition{Line: endLine, Character: endChar},
	}, nil
}
