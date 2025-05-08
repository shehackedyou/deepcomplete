// deepcomplete/types.go
// Contains core type definitions used throughout the deepcomplete package.
package deepcomplete

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	stdslog "log/slog"
	"net/url"
	"strings"
	"time"

	"golang.org/x/tools/go/packages"
)

// =============================================================================
// Configuration Types
// =============================================================================

const (
	defaultOllamaURL = "http://localhost:11434"
	defaultModel     = "deepseek-coder-r2"
	// Standard prompt template used for completions.
	promptTemplate = `<s>[INST] <<SYS>>
You are an expert Go programming assistant.
Analyze the provided context (enclosing function/method, imports, scope variables, comments, code structure) and the preceding code snippet.
Complete the Go code accurately and concisely.
Output ONLY the raw Go code completion, without any markdown, explanations, or introductory text.
<</SYS>>

CONTEXT:
%s

CODE SNIPPET TO COMPLETE:
` + "```go\n%s\n```" + `
[/INST]`
	// FIM prompt template used for fill-in-the-middle tasks.
	fimPromptTemplate = `<s>[INST] <<SYS>>
You are an expert Go programming assistant performing a fill-in-the-middle task.
Analyze the provided context and the code surrounding the <MID> marker.
Insert Go code at the <MID> marker to logically connect the <PRE>fix and <SUF>fix code blocks.
Output ONLY the raw Go code completion for the middle part, without any markdown, explanations, or introductory text.
<</SYS>>

CONTEXT:
%s

CODE TO FILL:
<PRE>%s<MID>%s<SUF>
[/INST]`

	defaultMaxTokens      = 256            // Default maximum tokens for LLM response.
	DefaultStop           = "\n"           // Default stop sequence for LLM. Exported for CLI use.
	defaultTemperature    = 0.1            // Default sampling temperature for LLM.
	defaultLogLevel       = "info"         // Default log level.
	defaultConfigFileName = "config.json"  // Default config file name.
	configDirName         = "deepcomplete" // Subdirectory name for config/data.
	cacheSchemaVersion    = 2              // Used to invalidate cache if internal formats change.

	// Retry constants
	maxRetries = 3
	retryDelay = 500 * time.Millisecond
)

// Config holds the active configuration for the autocompletion service.
type Config struct {
	OllamaURL      string   `json:"ollama_url"`
	Model          string   `json:"model"`
	PromptTemplate string   `json:"-"` // Loaded internally, not from config file.
	FimTemplate    string   `json:"-"` // Loaded internally, not from config file.
	MaxTokens      int      `json:"max_tokens"`
	Stop           []string `json:"stop"`
	Temperature    float64  `json:"temperature"`
	LogLevel       string   `json:"log_level"`        // Log level (debug, info, warn, error).
	UseAst         bool     `json:"use_ast"`          // Enable AST/Type analysis.
	UseFim         bool     `json:"use_fim"`          // Use Fill-in-the-Middle prompting.
	MaxPreambleLen int      `json:"max_preamble_len"` // Max bytes for AST context preamble.
	MaxSnippetLen  int      `json:"max_snippet_len"`  // Max bytes for code snippet context.
}

// FileConfig represents the structure of the JSON config file for unmarshalling.
// Uses pointers to distinguish between unset fields and zero-value fields.
type FileConfig struct {
	OllamaURL      *string   `json:"ollama_url"`
	Model          *string   `json:"model"`
	MaxTokens      *int      `json:"max_tokens"`
	Stop           *[]string `json:"stop"`
	Temperature    *float64  `json:"temperature"`
	LogLevel       *string   `json:"log_level"`
	UseAst         *bool     `json:"use_ast"`
	UseFim         *bool     `json:"use_fim"`
	MaxPreambleLen *int      `json:"max_preamble_len"`
	MaxSnippetLen  *int      `json:"max_snippet_len"`
}

// getDefaultConfig returns a new instance of the default configuration.
func getDefaultConfig() Config {
	return Config{
		OllamaURL:      defaultOllamaURL,
		Model:          defaultModel,
		PromptTemplate: promptTemplate,
		FimTemplate:    fimPromptTemplate,
		MaxTokens:      defaultMaxTokens,
		Stop:           []string{DefaultStop, "}", "//", "/*"}, // Sensible defaults
		Temperature:    defaultTemperature,
		LogLevel:       defaultLogLevel,
		UseAst:         true,  // Enable analysis by default
		UseFim:         false, // FIM is off by default
		MaxPreambleLen: 2048,  // Limit context preamble size
		MaxSnippetLen:  2048,  // Limit code snippet size
	}
}

// Validate checks if configuration values are valid, applying defaults for some fields.
// It uses the provided logger to report warnings about applied defaults.
func (c *Config) Validate(logger *stdslog.Logger) error {
	var validationErrors []error
	if logger == nil {
		logger = stdslog.Default() // Use default if nil
	}
	tempDefault := getDefaultConfig()

	if strings.TrimSpace(c.OllamaURL) == "" {
		validationErrors = append(validationErrors, errors.New("ollama_url cannot be empty"))
	} else {
		parsedURL, err := url.ParseRequestURI(c.OllamaURL)
		if err != nil {
			validationErrors = append(validationErrors, fmt.Errorf("invalid ollama_url format: %w", err))
		} else if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
			validationErrors = append(validationErrors, fmt.Errorf("invalid ollama_url scheme '%s', must be http or https", parsedURL.Scheme))
		}
	}
	if strings.TrimSpace(c.Model) == "" {
		validationErrors = append(validationErrors, errors.New("model cannot be empty"))
	}
	if c.MaxTokens <= 0 {
		logger.Warn("Config validation: max_tokens is not positive, applying default.", "configured_value", c.MaxTokens, "default", tempDefault.MaxTokens)
		c.MaxTokens = tempDefault.MaxTokens
	}
	// Validate Temperature range [0.0, 2.0]
	if c.Temperature < 0.0 || c.Temperature > 2.0 {
		logger.Warn("Config validation: temperature is outside reasonable range [0.0, 2.0], applying default.", "configured_value", c.Temperature, "default", tempDefault.Temperature)
		validationErrors = append(validationErrors, fmt.Errorf("temperature %f is outside valid range [0.0, 2.0]", c.Temperature))
		c.Temperature = tempDefault.Temperature // Apply default if out of range
	}
	if c.MaxPreambleLen <= 0 {
		logger.Warn("Config validation: max_preamble_len is not positive, applying default.", "configured_value", c.MaxPreambleLen, "default", tempDefault.MaxPreambleLen)
		c.MaxPreambleLen = tempDefault.MaxPreambleLen
	}
	if c.MaxSnippetLen <= 0 {
		logger.Warn("Config validation: max_snippet_len is not positive, applying default.", "configured_value", c.MaxSnippetLen, "default", tempDefault.MaxSnippetLen)
		c.MaxSnippetLen = tempDefault.MaxSnippetLen
	}
	if c.LogLevel == "" {
		logger.Warn("Config validation: log_level is empty, applying default.", "default", defaultLogLevel)
		c.LogLevel = defaultLogLevel
	} else {
		_, err := ParseLogLevel(c.LogLevel)
		if err != nil {
			logger.Warn("Config validation: Invalid log_level found, applying default.", "configured_value", c.LogLevel, "default", defaultLogLevel, "error", err)
			validationErrors = append(validationErrors, fmt.Errorf("invalid log_level '%s': %w", c.LogLevel, err))
			c.LogLevel = defaultLogLevel // Apply default if invalid
		}
	}
	if c.Stop == nil {
		logger.Warn("Config validation: stop sequences list is nil, applying default.", "default", tempDefault.Stop)
		c.Stop = make([]string, len(tempDefault.Stop))
		copy(c.Stop, tempDefault.Stop)
	}

	if c.PromptTemplate == "" {
		c.PromptTemplate = promptTemplate
	}
	if c.FimTemplate == "" {
		c.FimTemplate = fimPromptTemplate
	}

	if len(validationErrors) > 0 {
		return fmt.Errorf("%w: %w", ErrInvalidConfig, errors.Join(validationErrors...))
	}
	return nil
}

// =============================================================================
// Analysis & Context Types
// =============================================================================

type DiagnosticSeverity int

const (
	SeverityError   DiagnosticSeverity = 1
	SeverityWarning DiagnosticSeverity = 2
	SeverityInfo    DiagnosticSeverity = 3
	SeverityHint    DiagnosticSeverity = 4
)

type Position struct {
	Line      int // 0-based
	Character int // 0-based, byte offset within the line
}

type Range struct {
	Start Position
	End   Position
}

type Diagnostic struct {
	Range    Range
	Severity DiagnosticSeverity
	Code     string // Optional code for the diagnostic
	Source   string // e.g., "go", "deepcomplete-analyzer"
	Message  string
}

// AstContextInfo holds structured information extracted from code analysis.
type AstContextInfo struct {
	FilePath           string                  // Absolute path to the analyzed file.
	Version            int                     // Document version (for LSP).
	CursorPos          token.Pos               // Cursor position as a token.Pos.
	PackageName        string                  // Name of the package being analyzed.
	TargetPackage      *packages.Package       // Loaded package information.
	TargetFileSet      *token.FileSet          // FileSet used during loading.
	TargetAstFile      *ast.File               // AST of the specific file being analyzed.
	EnclosingFunc      *types.Func             // Type info for the enclosing function/method.
	EnclosingFuncNode  *ast.FuncDecl           // AST node for the enclosing function/method.
	ReceiverType       string                  // Formatted receiver type string if it's a method.
	EnclosingBlock     *ast.BlockStmt          // Innermost block statement containing the cursor.
	Imports            []*ast.ImportSpec       // List of imports in the file.
	CommentsNearCursor []string                // Relevant comments found near the cursor.
	IdentifierAtCursor *ast.Ident              // Identifier AST node at the cursor, if any.
	IdentifierType     types.Type              // Resolved type of the identifier at the cursor.
	IdentifierObject   types.Object            // Resolved object (var, func, type, etc.) for the identifier.
	IdentifierDefNode  ast.Node                // AST node where the identifier was defined (for hover/definition).
	SelectorExpr       *ast.SelectorExpr       // Selector expression (e.g., x.Y) enclosing the cursor.
	SelectorExprType   types.Type              // Resolved type of the base expression (x in x.Y).
	CallExpr           *ast.CallExpr           // Call expression enclosing the cursor.
	CallExprFuncType   *types.Signature        // Resolved type signature of the function being called.
	CallArgIndex       int                     // 0-based index of the argument the cursor is within.
	ExpectedArgType    types.Type              // Expected type of the argument at the cursor's position.
	CompositeLit       *ast.CompositeLit       // Composite literal (struct, slice, map) enclosing the cursor.
	CompositeLitType   types.Type              // Resolved type of the composite literal.
	VariablesInScope   map[string]types.Object // Map of identifiers (vars, consts, types, funcs) in scope.
	PromptPreamble     string                  // Generated context string for the LLM prompt.
	AnalysisErrors     []error                 // List of non-fatal errors encountered during analysis.
	Diagnostics        []Diagnostic            // List of diagnostics generated during analysis.
}

// =============================================================================
// Ollama & Cache Types
// =============================================================================

type OllamaError struct {
	Message string
	Status  int // HTTP status code, if available
}

func (e *OllamaError) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("Ollama error: %s (Status: %d)", e.Message, e.Status)
	}
	return fmt.Sprintf("Ollama error: %s", e.Message)
}

type OllamaResponse struct {
	Response string `json:"response"`        // The generated text chunk.
	Done     bool   `json:"done"`            // Indicates if the stream is complete.
	Error    string `json:"error,omitempty"` // Error message from Ollama, if any.
}

type CachedAnalysisData struct {
	PackageName    string // Cached package name.
	PromptPreamble string // Cached generated preamble.
}

type CachedAnalysisEntry struct {
	SchemaVersion   int               // Version of the cache structure itself.
	GoModHash       string            // Hash of the go.mod file when cached.
	InputFileHashes map[string]string // Hashes of relevant Go files (relative paths) when cached.
	AnalysisGob     []byte            // Gob-encoded CachedAnalysisData.
}

// =============================================================================
// Type Member Analysis Types
// =============================================================================

type MemberKind string

const (
	FieldMember  MemberKind = "field"
	MethodMember MemberKind = "method"
	OtherMember  MemberKind = "other" // Fallback
)

type MemberInfo struct {
	Name       string
	Kind       MemberKind
	TypeString string // Formatted type signature string.
}
