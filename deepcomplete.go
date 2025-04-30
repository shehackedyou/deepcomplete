// deepcomplete.go
// Package deepcomplete provides core logic for local code completion using LLMs.
package deepcomplete

import (
	"bytes"
	"context"
	"encoding/gob" // For cache serialization
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	stdslog "log/slog" // Use alias to avoid conflict if package name is slog
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/dgraph-io/ristretto"
	"go.etcd.io/bbolt"
	"golang.org/x/tools/go/packages"
)

// =============================================================================
// Constants & Core Types
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

var (
	cacheBucketName = []byte("AnalysisCache") // Name of the bbolt bucket for caching.
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

// Validate checks if configuration values are valid, applying defaults for some fields.
// It uses the default slog logger to report warnings about applied defaults.
func (c *Config) Validate() error {
	var validationErrors []error
	logger := stdslog.Default() // Use default logger

	if strings.TrimSpace(c.OllamaURL) == "" {
		validationErrors = append(validationErrors, errors.New("ollama_url cannot be empty"))
	} else if _, err := url.ParseRequestURI(c.OllamaURL); err != nil {
		validationErrors = append(validationErrors, fmt.Errorf("invalid ollama_url: %w", err))
	}
	if strings.TrimSpace(c.Model) == "" {
		validationErrors = append(validationErrors, errors.New("model cannot be empty"))
	}
	if c.MaxTokens <= 0 {
		logger.Warn("Config validation: max_tokens is not positive, applying default.", "configured_value", c.MaxTokens, "default", DefaultConfig.MaxTokens)
		c.MaxTokens = DefaultConfig.MaxTokens
	}
	if c.Temperature < 0 {
		logger.Warn("Config validation: temperature is negative, applying default.", "configured_value", c.Temperature, "default", DefaultConfig.Temperature)
		c.Temperature = DefaultConfig.Temperature
	}
	if c.MaxPreambleLen <= 0 {
		logger.Warn("Config validation: max_preamble_len is not positive, applying default.", "configured_value", c.MaxPreambleLen, "default", DefaultConfig.MaxPreambleLen)
		c.MaxPreambleLen = DefaultConfig.MaxPreambleLen
	}
	if c.MaxSnippetLen <= 0 {
		logger.Warn("Config validation: max_snippet_len is not positive, applying default.", "configured_value", c.MaxSnippetLen, "default", DefaultConfig.MaxSnippetLen)
		c.MaxSnippetLen = DefaultConfig.MaxSnippetLen
	}
	if c.LogLevel == "" {
		logger.Warn("Config validation: log_level is empty, applying default.", "default", defaultLogLevel)
		c.LogLevel = defaultLogLevel
	} else {
		_, err := ParseLogLevel(c.LogLevel) // Use helper to validate
		if err != nil {
			logger.Warn("Config validation: Invalid log_level found, applying default.", "configured_value", c.LogLevel, "default", defaultLogLevel, "error", err)
			c.LogLevel = defaultLogLevel
		}
	}
	// Stop sequences: Ensure it's not nil, but allow empty slice.
	if c.Stop == nil {
		logger.Warn("Config validation: stop sequences list is nil, applying default.", "default", DefaultConfig.Stop)
		c.Stop = make([]string, len(DefaultConfig.Stop))
		copy(c.Stop, DefaultConfig.Stop)
	}

	if len(validationErrors) > 0 {
		return errors.Join(validationErrors...)
	}
	return nil
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

// Diagnostic Structures (Internal representation used by analysis)
type DiagnosticSeverity int

const (
	SeverityError   DiagnosticSeverity = 1
	SeverityWarning DiagnosticSeverity = 2
	SeverityInfo    DiagnosticSeverity = 3
	SeverityHint    DiagnosticSeverity = 4
)

// Position represents a 0-based line and byte offset within that line.
type Position struct {
	Line      int // 0-based
	Character int // 0-based, byte offset within the line
}

// Range represents a span in the source code using internal Positions.
type Range struct {
	Start Position
	End   Position
}

// Diagnostic represents an issue found during analysis.
type Diagnostic struct {
	Range    Range
	Severity DiagnosticSeverity
	Code     string // Optional code for the diagnostic
	Source   string // e.g., "go", "deepcomplete-analyzer"
	Message  string
}

// AstContextInfo holds structured information extracted from code analysis.
// This structure is populated by the Analyzer and used by the PreambleFormatter and LSP handlers.
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

// OllamaError defines a custom error for Ollama API issues, including HTTP status.
type OllamaError struct {
	Message string
	Status  int // HTTP status code, if available
}

// Error implements the error interface for OllamaError.
func (e *OllamaError) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("Ollama error: %s (Status: %d)", e.Message, e.Status)
	}
	return fmt.Sprintf("Ollama error: %s", e.Message)
}

// OllamaResponse represents the streaming response structure from Ollama's /api/generate.
type OllamaResponse struct {
	Response string `json:"response"`        // The generated text chunk.
	Done     bool   `json:"done"`            // Indicates if the stream is complete.
	Error    string `json:"error,omitempty"` // Error message from Ollama, if any.
}

// CachedAnalysisData holds derived information stored in the bbolt cache (gob-encoded).
// This contains data that is expensive to compute but can be reused if inputs haven't changed.
type CachedAnalysisData struct {
	PackageName    string // Cached package name.
	PromptPreamble string // Cached generated preamble.
	// Add other derived data here if needed (e.g., serialized scope info)
}

// CachedAnalysisEntry represents the full structure stored in bbolt.
// It includes metadata to validate the cache entry against current file states.
type CachedAnalysisEntry struct {
	SchemaVersion   int               // Version of the cache structure itself.
	GoModHash       string            // Hash of the go.mod file when cached.
	InputFileHashes map[string]string // Hashes of relevant Go files (relative paths) when cached.
	AnalysisGob     []byte            // Gob-encoded CachedAnalysisData.
}

// MemberKind defines the type of member (field or method) for type analysis.
type MemberKind string

const (
	FieldMember  MemberKind = "field"
	MethodMember MemberKind = "method"
	OtherMember  MemberKind = "other" // Fallback
)

// MemberInfo holds structured information about a type member (field or method).
type MemberInfo struct {
	Name       string
	Kind       MemberKind
	TypeString string // Formatted type signature string.
}

// =============================================================================
// Exported Errors
// =============================================================================

var (
	// ErrAnalysisFailed indicates non-fatal errors occurred during code analysis.
	// Further details should be available in AstContextInfo.AnalysisErrors.
	ErrAnalysisFailed = errors.New("code analysis failed")
	// ErrOllamaUnavailable indicates failure communicating with the Ollama API.
	ErrOllamaUnavailable = errors.New("ollama API unavailable")
	// ErrStreamProcessing indicates an error reading or processing the LLM response stream.
	ErrStreamProcessing = errors.New("error processing LLM stream")
	// ErrConfig indicates non-fatal errors during config loading or processing.
	ErrConfig = errors.New("configuration error")
	// ErrInvalidConfig indicates a configuration value is invalid after validation.
	ErrInvalidConfig = errors.New("invalid configuration")
	// ErrCache indicates a general cache operation failure.
	ErrCache = errors.New("cache operation failed")
	// ErrCacheRead indicates failure reading from the cache.
	ErrCacheRead = errors.New("cache read failed")
	// ErrCacheWrite indicates failure writing to the cache.
	ErrCacheWrite = errors.New("cache write failed")
	// ErrCacheDecode indicates failure decoding data read from the cache.
	ErrCacheDecode = errors.New("cache decode failed")
	// ErrCacheEncode indicates failure encoding data for writing to the cache.
	ErrCacheEncode = errors.New("cache encode failed")
	// ErrCacheHash indicates failure calculating file hashes for cache validation.
	ErrCacheHash = errors.New("cache hash calculation failed")
	// ErrPositionConversion indicates failure converting between position formats (e.g., LSP <-> byte offset).
	ErrPositionConversion = errors.New("position conversion failed")
	// ErrInvalidPositionInput indicates input position values (line/col) are invalid.
	ErrInvalidPositionInput = errors.New("invalid input position")
	// ErrPositionOutOfRange indicates a position is outside the valid bounds of the file or line.
	ErrPositionOutOfRange = errors.New("position out of range")
	// ErrInvalidUTF8 indicates an invalid UTF-8 sequence was encountered during processing.
	ErrInvalidUTF8 = errors.New("invalid utf-8 sequence")
	// ErrInvalidURI indicates a document URI is invalid or uses an unsupported scheme.
	ErrInvalidURI = errors.New("invalid document URI")
)

// =============================================================================
// Interfaces for Components
// =============================================================================

// LLMClient defines the interface for interacting with the language model backend.
type LLMClient interface {
	// GenerateStream sends a prompt to the LLM and returns a stream of generated text.
	GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error)
}

// Analyzer defines the interface for code analysis.
type Analyzer interface {
	// Analyze performs static analysis on the given file at the specified position.
	Analyze(ctx context.Context, filename string, version int, line, col int) (*AstContextInfo, error)
	// Close cleans up any resources used by the analyzer (e.g., cache connections).
	Close() error
	// InvalidateCache removes cached data related to a specific directory (e.g., when go.mod changes).
	InvalidateCache(dir string) error
	// InvalidateMemoryCacheForURI removes memory-cached data for a specific file URI (e.g., on file change).
	InvalidateMemoryCacheForURI(uri string, version int) error
	// MemoryCacheEnabled returns true if the in-memory cache is configured and active.
	MemoryCacheEnabled() bool
	// GetMemoryCacheMetrics returns performance metrics for the in-memory cache.
	GetMemoryCacheMetrics() *ristretto.Metrics
}

// PromptFormatter defines the interface for constructing the final prompt sent to the LLM.
type PromptFormatter interface {
	// FormatPrompt combines the analysis preamble and code snippet context into the final prompt string.
	FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string
}

// =============================================================================
// Variables & Default Config
// =============================================================================

var (
	// DefaultConfig provides default settings used when no config file is found or fields are missing.
	DefaultConfig = Config{
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
)

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadConfig loads configuration from standard locations (XDG_CONFIG_HOME, ~/.config),
// merges with defaults, validates the result, and attempts to write a default config
// if none exists or the existing one is invalid.
// Returns the final validated Config and a non-fatal ErrConfig if warnings occurred during loading.
func LoadConfig() (Config, error) {
	logger := stdslog.Default() // Use default logger, assuming it's set up by main
	cfg := DefaultConfig        // Start with defaults
	var loadedFromFile bool
	var loadErrors []error
	var configParseError error

	// Determine primary (XDG) and secondary (~/.config) paths
	primaryPath, secondaryPath, pathErr := getConfigPaths()
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		logger.Warn("Could not determine config paths, will use defaults", "error", pathErr)
	}

	// Try loading from primary path
	if primaryPath != "" {
		logger.Debug("Attempting to load config", "path", primaryPath)
		loaded, loadErr := loadAndMergeConfig(primaryPath, &cfg, logger)
		if loadErr != nil {
			// Check if it's specifically a JSON parsing error
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr // Store the parsing error
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", primaryPath, loadErr))
			logger.Warn("Failed to load or merge config", "path", primaryPath, "error", loadErr)
		} else if loaded {
			loadedFromFile = true
			logger.Info("Loaded config", "path", primaryPath)
		} else {
			logger.Debug("Config file not found or empty", "path", primaryPath)
		}
	}

	// Try secondary path if primary failed or wasn't found
	primaryNotFoundOrFailed := !loadedFromFile || configParseError != nil
	if primaryNotFoundOrFailed && secondaryPath != "" {
		logger.Debug("Attempting to load config from secondary path", "path", secondaryPath)
		loaded, loadErr := loadAndMergeConfig(secondaryPath, &cfg, logger)
		if loadErr != nil {
			// Store parsing error only if we didn't already have one from the primary path
			if configParseError == nil && strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", secondaryPath, loadErr))
			logger.Warn("Failed to load or merge config", "path", secondaryPath, "error", loadErr)
		} else if loaded {
			// Only mark as loadedFromFile if we hadn't already loaded from primary
			if !loadedFromFile {
				loadedFromFile = true
				logger.Info("Loaded config", "path", secondaryPath)
			}
		} else {
			logger.Debug("Config file not found or empty", "path", secondaryPath)
		}
	}

	// Write default config if no valid file was loaded
	loadSucceeded := loadedFromFile && configParseError == nil
	if !loadSucceeded {
		if configParseError != nil {
			logger.Warn("Existing config file failed to parse. Attempting to write default.", "error", configParseError)
		} else {
			logger.Info("No valid config file found. Attempting to write default.")
		}
		// Determine path to write default config (prefer primary)
		writePath := primaryPath
		if writePath == "" {
			writePath = secondaryPath
		}

		if writePath != "" {
			logger.Info("Attempting to write default config", "path", writePath)
			if err := writeDefaultConfig(writePath, DefaultConfig); err != nil {
				logger.Warn("Failed to write default config", "path", writePath, "error", err)
				loadErrors = append(loadErrors, fmt.Errorf("writing default config failed: %w", err))
			}
		} else {
			logger.Warn("Cannot determine path to write default config.")
			loadErrors = append(loadErrors, errors.New("cannot determine default config path"))
		}
		cfg = DefaultConfig // Reset to defaults if write failed or no path found
		logger.Info("Using default configuration values.")
	}

	// Ensure internal templates are set (they aren't loaded from file)
	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	// Final validation of the resulting config (loaded or default)
	finalCfg := cfg
	if err := finalCfg.Validate(); err != nil {
		logger.Warn("Config after load/merge failed validation. Falling back to pure defaults.", "error", err)
		loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		// Validate the pure DefaultConfig as a last resort
		if valErr := DefaultConfig.Validate(); valErr != nil {
			// This indicates a bug in the DefaultConfig definition
			logger.Error("FATAL: Default config is invalid", "error", valErr)
			return DefaultConfig, fmt.Errorf("default config is invalid: %w", valErr)
		}
		finalCfg = DefaultConfig // Use pure defaults if merged/loaded config is invalid
	}

	// Return final config and wrap any accumulated non-fatal errors
	if len(loadErrors) > 0 {
		return finalCfg, fmt.Errorf("%w: %w", ErrConfig, errors.Join(loadErrors...))
	}
	return finalCfg, nil
}

// getConfigPaths determines the primary (XDG_CONFIG_HOME) and secondary (~/.config) config paths.
func getConfigPaths() (primary string, secondary string, err error) {
	var cfgErr, homeErr error
	logger := stdslog.Default() // Use default logger

	// Primary Path: XDG_CONFIG_HOME
	userConfigDir, cfgErr := os.UserConfigDir()
	if cfgErr == nil {
		primary = filepath.Join(userConfigDir, configDirName, defaultConfigFileName)
	} else {
		logger.Warn("Could not determine user config directory (XDG)", "error", cfgErr)
	}

	// Secondary Path: ~/.config
	homeDir, homeErr := os.UserHomeDir()
	if homeErr == nil {
		secondary = filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
		// If primary path failed, use secondary as primary
		if primary == "" && cfgErr != nil {
			primary = secondary
			logger.Debug("Using fallback primary config path", "path", primary)
			secondary = "" // Avoid duplication
		}
		// If paths ended up the same, clear secondary
		if primary == secondary {
			secondary = ""
		}
	} else {
		logger.Warn("Could not determine user home directory", "error", homeErr)
	}

	// If neither path could be determined, return an error
	if primary == "" && secondary == "" {
		err = fmt.Errorf("cannot determine config/home directories: config error: %v; home error: %v", cfgErr, homeErr)
	}
	return primary, secondary, err
}

// loadAndMergeConfig attempts to load config from a specific path and merge its
// fields into the provided cfg object.
// Returns true if the file was found and read (even if empty or unparsable), false otherwise.
// Returns an error if reading or parsing fails.
func loadAndMergeConfig(path string, cfg *Config, logger *stdslog.Logger) (loaded bool, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil // File not found is not an error here
		}
		return false, fmt.Errorf("reading config file %q failed: %w", path, err)
	}

	// File exists, mark as loaded
	loaded = true

	if len(data) == 0 {
		logger.Warn("Config file exists but is empty, ignoring.", "path", path)
		return loaded, nil // Empty file is not a parsing error
	}

	var fileCfg FileConfig
	if err := json.Unmarshal(data, &fileCfg); err != nil {
		return loaded, fmt.Errorf("parsing config file JSON %q failed: %w", path, err)
	}

	// Merge fields from fileCfg into cfg if they are non-nil
	mergedFields := 0
	if fileCfg.OllamaURL != nil {
		cfg.OllamaURL = *fileCfg.OllamaURL
		mergedFields++
	}
	if fileCfg.Model != nil {
		cfg.Model = *fileCfg.Model
		mergedFields++
	}
	if fileCfg.MaxTokens != nil {
		cfg.MaxTokens = *fileCfg.MaxTokens
		mergedFields++
	}
	if fileCfg.Stop != nil {
		cfg.Stop = *fileCfg.Stop
		mergedFields++
	}
	if fileCfg.Temperature != nil {
		cfg.Temperature = *fileCfg.Temperature
		mergedFields++
	}
	if fileCfg.LogLevel != nil {
		cfg.LogLevel = *fileCfg.LogLevel
		mergedFields++
	}
	if fileCfg.UseAst != nil {
		cfg.UseAst = *fileCfg.UseAst
		mergedFields++
	}
	if fileCfg.UseFim != nil {
		cfg.UseFim = *fileCfg.UseFim
		mergedFields++
	}
	if fileCfg.MaxPreambleLen != nil {
		cfg.MaxPreambleLen = *fileCfg.MaxPreambleLen
		mergedFields++
	}
	if fileCfg.MaxSnippetLen != nil {
		cfg.MaxSnippetLen = *fileCfg.MaxSnippetLen
		mergedFields++
	}
	logger.Debug("Merged configuration from file", "path", path, "fields_merged", mergedFields)

	return loaded, nil
}

// writeDefaultConfig creates the config directory if needed and writes the
// default configuration values as a JSON file.
func writeDefaultConfig(path string, defaultConfig Config) error {
	logger := stdslog.Default() // Use default logger
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0750); err != nil { // Use appropriate permissions
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}

	// Create an exportable struct containing only fields meant for the config file
	type ExportableConfig struct {
		OllamaURL      string   `json:"ollama_url"`
		Model          string   `json:"model"`
		MaxTokens      int      `json:"max_tokens"`
		Stop           []string `json:"stop"`
		Temperature    float64  `json:"temperature"`
		LogLevel       string   `json:"log_level"`
		UseAst         bool     `json:"use_ast"`
		UseFim         bool     `json:"use_fim"`
		MaxPreambleLen int      `json:"max_preamble_len"`
		MaxSnippetLen  int      `json:"max_snippet_len"`
	}
	expCfg := ExportableConfig{
		OllamaURL:      defaultConfig.OllamaURL,
		Model:          defaultConfig.Model,
		MaxTokens:      defaultConfig.MaxTokens,
		Stop:           defaultConfig.Stop,
		Temperature:    defaultConfig.Temperature,
		LogLevel:       defaultConfig.LogLevel,
		UseAst:         defaultConfig.UseAst,
		UseFim:         defaultConfig.UseFim,
		MaxPreambleLen: defaultConfig.MaxPreambleLen,
		MaxSnippetLen:  defaultConfig.MaxSnippetLen,
	}

	jsonData, err := json.MarshalIndent(expCfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal default config to JSON: %w", err)
	}

	// Write with restricted permissions
	if err := os.WriteFile(path, jsonData, 0640); err != nil {
		return fmt.Errorf("failed to write default config file %s: %w", path, err)
	}
	logger.Info("Wrote default configuration", "path", path)
	return nil
}

// ParseLogLevel converts a log level string to its corresponding slog.Level constant.
func ParseLogLevel(levelStr string) (stdslog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(levelStr)) {
	case "debug":
		return stdslog.LevelDebug, nil
	case "info":
		return stdslog.LevelInfo, nil
	case "warn", "warning":
		return stdslog.LevelWarn, nil
	case "error", "err":
		return stdslog.LevelError, nil
	default:
		return stdslog.LevelInfo, fmt.Errorf("invalid log level string: %q (expected debug, info, warn, or error)", levelStr)
	}
}

// =============================================================================
// Default Component Implementations
// =============================================================================

// httpOllamaClient implements the LLMClient interface using HTTP requests to an Ollama server.
type httpOllamaClient struct {
	httpClient *http.Client
}

// newHttpOllamaClient creates a new Ollama client with reasonable timeouts.
func newHttpOllamaClient() *httpOllamaClient {
	return &httpOllamaClient{
		httpClient: &http.Client{
			Timeout: 90 * time.Second, // Overall request timeout
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: 10 * time.Second, // Connection timeout
				}).DialContext,
				TLSHandshakeTimeout: 10 * time.Second,
				MaxIdleConns:        10,
				IdleConnTimeout:     30 * time.Second,
			},
		},
	}
}

// GenerateStream sends a request to Ollama's /api/generate endpoint and returns the streaming response body.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	logger := stdslog.Default() // Use default logger
	base := strings.TrimSuffix(config.OllamaURL, "/")
	endpointURL := base + "/api/generate"
	u, err := url.Parse(endpointURL)
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL '%s': %w", endpointURL, err)
	}

	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true, // Ensure streaming is enabled
		"options": map[string]interface{}{
			"temperature": config.Temperature,
			"num_ctx":     4096, // Consider making this configurable if needed
			"top_p":       0.9,  // Common default, consider making configurable
			"stop":        config.Stop,
			"num_predict": config.MaxTokens,
		},
	}
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/x-ndjson") // Ollama uses newline-delimited JSON for streaming

	logger.Debug("Sending request to Ollama", "url", endpointURL, "model", config.Model)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Handle context cancellation/deadline explicitly
		if errors.Is(err, context.Canceled) {
			logger.Warn("Ollama request cancelled", "url", endpointURL)
			return nil, context.Canceled
		}
		if errors.Is(err, context.DeadlineExceeded) {
			logger.Error("Ollama request timed out", "url", endpointURL, "timeout", c.httpClient.Timeout)
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		// Handle network errors
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			logger.Error("Network timeout connecting to Ollama", "host", u.Host)
			return nil, fmt.Errorf("%w: network timeout connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		// Handle connection refused specifically
		if opErr, ok := err.(*net.OpError); ok && opErr.Op == "dial" {
			logger.Error("Connection refused or network error connecting to Ollama", "host", u.Host)
			return nil, fmt.Errorf("%w: connection refused or network error connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
		}
		// General HTTP request failure
		logger.Error("HTTP request to Ollama failed", "url", endpointURL, "error", err)
		return nil, fmt.Errorf("%w: http request failed: %w", ErrOllamaUnavailable, err)
	}

	// Check for non-200 status codes
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(failed to read error response body)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			// Try to parse Ollama's specific error structure
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error // Use the specific error message if available
			}
		}
		err = &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		logger.Error("Ollama API returned non-OK status", "status", resp.Status, "response_body", bodyString)
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
	}

	// Return the response body for streaming
	return resp.Body, nil
}

// GoPackagesAnalyzer implements the Analyzer interface using go/packages and bbolt/ristretto caching.
type GoPackagesAnalyzer struct {
	db          *bbolt.DB        // Persistent disk cache (bbolt)
	memoryCache *ristretto.Cache // In-memory cache (ristretto)
	mu          sync.Mutex       // Protects access to db/memoryCache handles during Close/Invalidate
}

// NewGoPackagesAnalyzer initializes the analyzer, including setting up bbolt and ristretto caches.
func NewGoPackagesAnalyzer() *GoPackagesAnalyzer {
	logger := stdslog.Default() // Use default logger
	dbPath := ""

	// Determine cache directory path
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		// Include schema version in the path to automatically invalidate old caches
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil { // Ensure directory exists with appropriate permissions
			dbPath = filepath.Join(dbDir, "analysis_cache.db")
		} else {
			logger.Warn("Could not create bbolt cache directory, disk caching disabled.", "path", dbDir, "error", err)
		}
	} else {
		logger.Warn("Could not determine user cache directory, disk caching disabled.", "error", err)
	}

	// Initialize bbolt database if path is valid
	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{Timeout: 1 * time.Second} // Set a reasonable timeout
		db, err = bbolt.Open(dbPath, 0600, opts)         // Use restricted file permissions
		if err != nil {
			logger.Warn("Failed to open bbolt cache file, disk caching disabled.", "path", dbPath, "error", err)
			db = nil // Ensure db is nil if open fails
		} else {
			// Ensure the cache bucket exists
			err = db.Update(func(tx *bbolt.Tx) error {
				_, err := tx.CreateBucketIfNotExists(cacheBucketName)
				if err != nil {
					return fmt.Errorf("failed to create cache bucket %s: %w", string(cacheBucketName), err)
				}
				return nil
			})
			if err != nil {
				logger.Warn("Failed to ensure bbolt bucket exists, disk caching disabled.", "error", err)
				db.Close() // Close the db if bucket creation failed
				db = nil
			} else {
				logger.Info("Using bbolt disk cache", "path", dbPath, "schema_version", cacheSchemaVersion)
			}
		}
	}

	// Initialize Ristretto memory cache
	memCache, cacheErr := ristretto.NewCache(&ristretto.Config{
		NumCounters: 1e7,     // 10M keys recommended by Ristretto docs
		MaxCost:     1 << 30, // 1GB max cache size
		BufferItems: 64,      // Default internal buffer size
		Metrics:     true,    // Enable metrics collection
	})
	if cacheErr != nil {
		logger.Warn("Failed to create ristretto memory cache, in-memory caching disabled.", "error", cacheErr)
		memCache = nil // Ensure memCache is nil if creation fails
	} else {
		logger.Info("Initialized ristretto in-memory cache", "max_cost", "1GB")
	}

	return &GoPackagesAnalyzer{db: db, memoryCache: memCache}
}

// Close cleans up resources used by the analyzer, primarily closing cache connections.
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock() // Protect access to db/memoryCache handles
	defer a.mu.Unlock()
	var closeErrors []error
	logger := stdslog.Default()

	if a.db != nil {
		logger.Info("Closing bbolt cache database.")
		if err := a.db.Close(); err != nil {
			logger.Error("Error closing bbolt database", "error", err)
			closeErrors = append(closeErrors, fmt.Errorf("bbolt close failed: %w", err))
		}
		a.db = nil // Mark as closed
	}
	if a.memoryCache != nil {
		logger.Info("Closing ristretto memory cache.")
		a.memoryCache.Close()
		a.memoryCache = nil // Mark as closed
	}

	if len(closeErrors) > 0 {
		return errors.Join(closeErrors...)
	}
	return nil
}

// Analyze performs code analysis for a given file and position, utilizing caching.
// It orchestrates calls to loading, analysis steps, and preamble generation helpers.
// Returns the populated AstContextInfo and any fatal error encountered during loading/analysis.
// Non-fatal errors are collected within AstContextInfo.AnalysisErrors.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, absFilename string, version int, line, col int) (info *AstContextInfo, analysisErr error) {
	logger := stdslog.Default().With("absFile", absFilename, "version", version, "line", line, "col", col)
	// Initialize the result struct
	info = &AstContextInfo{
		FilePath:         absFilename,
		Version:          version,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		Diagnostics:      make([]Diagnostic, 0),
		CallArgIndex:     -1, // Initialize to -1 (no call context initially)
	}

	// Panic recovery defer function
	defer func() {
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			logger.Error("Panic recovered during Analyze", "error", r, "stack", string(debug.Stack()))
			addAnalysisError(info, panicErr, logger) // Add panic as a non-fatal analysis error
			// Ensure analysisErr reflects the panic if no other fatal error occurred
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr)
			}
		}
		// If non-fatal errors occurred but no fatal error was returned, wrap them in ErrAnalysisFailed
		if len(info.AnalysisErrors) > 0 && analysisErr == nil {
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", ErrAnalysisFailed, finalErr)
		} else if len(info.AnalysisErrors) > 0 && analysisErr != nil {
			// If a fatal error already exists, join the non-fatal ones to it
			finalErr := errors.Join(info.AnalysisErrors...)
			analysisErr = fmt.Errorf("%w: %w", analysisErr, finalErr)
		}
	}()

	logger.Info("Starting context analysis")
	dir := filepath.Dir(absFilename)
	goModHash := calculateGoModHash(dir) // Helper from utils
	cacheKey := []byte(dir + "::" + goModHash)
	cacheHit := false
	var cachedEntry *CachedAnalysisEntry
	var loadDuration, stepsDuration, preambleDuration time.Duration

	// --- Bbolt Cache Check ---
	if a.db != nil {
		readStart := time.Now()
		dbViewErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil {
				logger.Debug("Bbolt cache bucket not found.")
				return nil // Not an error, just no bucket
			}
			valBytes := b.Get(cacheKey)
			if valBytes == nil {
				logger.Debug("Bbolt cache miss.", "key", string(cacheKey))
				return nil // Cache miss
			}

			logger.Debug("Bbolt cache hit (raw bytes found). Decoding entry...", "key", string(cacheKey))
			var decoded CachedAnalysisEntry
			// Use gob to decode the full entry structure
			if err := gob.NewDecoder(bytes.NewReader(valBytes)).Decode(&decoded); err != nil {
				// Wrap decode error for clarity
				return fmt.Errorf("%w: %w", ErrCacheDecode, err)
			}
			// Validate schema version
			if decoded.SchemaVersion != cacheSchemaVersion {
				logger.Warn("Bbolt cache data has old schema version. Ignoring.", "key", string(cacheKey), "cached_version", decoded.SchemaVersion, "expected_version", cacheSchemaVersion)
				return nil // Treat as miss due to schema mismatch
			}
			cachedEntry = &decoded // Store decoded entry
			return nil
		})
		// Handle errors from the View operation
		if dbViewErr != nil {
			logger.Warn("Error reading or decoding from bbolt cache. Cache check failed.", "error", dbViewErr)
			addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheRead, dbViewErr), logger)
			// If decoding failed, invalidate the bad entry
			if errors.Is(dbViewErr, ErrCacheDecode) {
				go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "decode_failure"))
			}
			cachedEntry = nil // Ensure entry is nil on error
		}
		logger.Debug("Bbolt cache read attempt finished", "duration", time.Since(readStart))

		// --- Bbolt Cache Validation ---
		if cachedEntry != nil {
			validationStart := time.Now()
			logger.Debug("Potential bbolt cache hit. Validating file hashes...", "key", string(cacheKey))
			// Calculate current hashes (passing nil pkg as we don't have it yet)
			currentHashes, hashErr := calculateInputHashes(dir, nil) // Helper from utils
			// Check go.mod hash and compare file hashes
			if hashErr == nil && cachedEntry.GoModHash == goModHash && compareFileHashes(currentHashes, cachedEntry.InputFileHashes) {
				logger.Debug("Bbolt cache VALID. Attempting to decode analysis data...", "key", string(cacheKey))
				decodeStart := time.Now()
				var analysisData CachedAnalysisData
				// Decode the gob-encoded analysis data
				if decodeErr := gob.NewDecoder(bytes.NewReader(cachedEntry.AnalysisGob)).Decode(&analysisData); decodeErr == nil {
					// Populate info struct from cached data
					info.PackageName = analysisData.PackageName
					info.PromptPreamble = analysisData.PromptPreamble
					cacheHit = true
					loadDuration = time.Since(decodeStart) // Use decode time as effective load time
					logger.Debug("Analysis data successfully decoded from bbolt cache.", "duration", loadDuration)
					// NOTE: Even with a cache hit for preamble, we might still need to run
					// analysis steps to get diagnostics or hover info.
					// For simplicity now, we force re-analysis if not just getting preamble.
					// A more advanced cache could store more analysis results.
					logger.Debug("Re-running load/analysis steps for diagnostics/hover despite cache hit.")
					cacheHit = false // Force re-analysis for now
				} else {
					logger.Warn("Failed to gob-decode cached analysis data. Treating as miss.", "error", decodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheDecode, decodeErr), logger)
					go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "analysis_decode_failure"))
					cacheHit = false
				}
			} else {
				// Cache is invalid (hash mismatch or error calculating current hashes)
				logger.Debug("Bbolt cache INVALID (hash mismatch or error). Treating as miss.", "key", string(cacheKey), "hash_error", hashErr)
				go deleteCacheEntryByKey(a.db, cacheKey, logger.With("reason", "hash_mismatch"))
				if hashErr != nil {
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger)
				}
				cacheHit = false
			}
			logger.Debug("Bbolt cache validation/decode finished", "duration", time.Since(validationStart))
		}
	} else {
		logger.Debug("Bbolt cache disabled (db handle is nil).")
	}

	// --- Perform Full Analysis if Cache Miss or Invalid ---
	if !cacheHit {
		logger.Debug("Cache miss or invalid (or re-running for diagnostics). Performing full analysis...", "key", string(cacheKey))
		loadStart := time.Now()
		fset := token.NewFileSet()
		info.TargetFileSet = fset // Store fileset in context

		// Load package and file information
		var loadDiagnostics []Diagnostic
		var loadErrors []error
		targetPkg, targetFileAST, targetFile, loadDiagnostics, loadErrors := loadPackageAndFile(ctx, absFilename, fset, logger) // From helpers_loader.go
		info.Diagnostics = append(info.Diagnostics, loadDiagnostics...)                                                         // Append diagnostics from loading phase

		loadDuration = time.Since(loadStart)
		logger.Debug("packages.Load completed", "duration", loadDuration)
		// Add loading errors to the context
		for _, loadErr := range loadErrors {
			addAnalysisError(info, loadErr, logger)
			// Diagnostics were already created in loadPackageAndFile
		}
		// Store loaded package and AST in context
		info.TargetPackage = targetPkg
		info.TargetAstFile = targetFileAST

		// Perform detailed analysis steps only if loading was somewhat successful
		stepsStart := time.Now()
		if targetFile != nil { // Need the token.File for position calculations
			analyzeStepErr := performAnalysisSteps(targetFile, targetFileAST, targetPkg, fset, line, col, a, info, logger) // From helpers_analysis_steps.go
			if analyzeStepErr != nil {
				// This function adds errors to info internally, but return fatal ones
				addAnalysisError(info, analyzeStepErr, logger) // Log/store potentially fatal error
			}
		} else {
			// If targetFile is nil, likely due to critical load errors
			if len(loadErrors) == 0 { // Add error only if not already reported by loader
				addAnalysisError(info, errors.New("cannot perform analysis steps: target token.File is nil"), logger)
			}
			// Attempt to gather package scope even without file/AST context
			gatherScopeContext(nil, targetPkg, fset, info, logger) // From helpers_analysis_steps.go
		}
		stepsDuration = time.Since(stepsStart)
		logger.Debug("Analysis steps completed", "duration", stepsDuration)

		// Construct the preamble if not loaded from cache
		// (In current logic, preamble is always rebuilt if cacheHit is false)
		if info.PromptPreamble == "" {
			preambleStart := time.Now()
			var qualifier types.Qualifier // Qualifier for formatting type names relative to the package
			if targetPkg != nil && targetPkg.Types != nil {
				if info.PackageName == "" { // Set package name if not already set
					info.PackageName = targetPkg.Types.Name()
				}
				qualifier = types.RelativeTo(targetPkg.Types)
			} else {
				// Fallback qualifier if type info is missing
				qualifier = func(other *types.Package) string {
					if other != nil {
						return other.Path() // Use full package path as fallback
					}
					return ""
				}
				logger.Debug("Building preamble with limited or no type info.")
			}
			// Generate the preamble string
			info.PromptPreamble = constructPromptPreamble(a, info, qualifier, logger) // From helpers_preamble.go
			preambleDuration = time.Since(preambleStart)
			logger.Debug("Preamble construction completed", "duration", preambleDuration)
		} else {
			logger.Debug("Skipping preamble construction (loaded from cache or already built).")
		}

		// --- Bbolt Cache Write ---
		// Save results to cache if enabled, analysis succeeded (no fatal load errors), and preamble was generated
		shouldSave := a.db != nil && info.PromptPreamble != "" && len(loadErrors) == 0
		if shouldSave {
			logger.Debug("Attempting to save analysis results (preamble) to bbolt cache.", "key", string(cacheKey))
			saveStart := time.Now()
			// Calculate input file hashes for validation
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg) // Helper from utils
			if hashErr == nil {
				// Prepare data for caching
				analysisDataToCache := CachedAnalysisData{
					PackageName:    info.PackageName,
					PromptPreamble: info.PromptPreamble,
				}
				var gobBuf bytes.Buffer
				// Encode analysis data using gob
				if encodeErr := gob.NewEncoder(&gobBuf).Encode(&analysisDataToCache); encodeErr == nil {
					analysisGob := gobBuf.Bytes()
					// Create the full cache entry structure
					entryToSave := CachedAnalysisEntry{
						SchemaVersion:   cacheSchemaVersion,
						GoModHash:       goModHash,
						InputFileHashes: inputHashes,
						AnalysisGob:     analysisGob,
					}
					var entryBuf bytes.Buffer
					// Encode the full entry using gob
					if entryEncodeErr := gob.NewEncoder(&entryBuf).Encode(&entryToSave); entryEncodeErr == nil {
						encodedBytes := entryBuf.Bytes()
						// Write the encoded entry to bbolt
						saveErr := a.db.Update(func(tx *bbolt.Tx) error {
							b := tx.Bucket(cacheBucketName)
							if b == nil {
								// This shouldn't happen if initialization was correct
								return fmt.Errorf("%w: cache bucket %s disappeared", ErrCacheWrite, string(cacheBucketName))
							}
							logger.Debug("Writing bytes to bbolt cache", "key", string(cacheKey), "bytes", len(encodedBytes))
							return b.Put(cacheKey, encodedBytes)
						})
						if saveErr == nil {
							logger.Debug("Saved analysis results (preamble) to bbolt cache", "key", string(cacheKey), "duration", time.Since(saveStart))
						} else {
							logger.Warn("Failed to write to bbolt cache", "key", string(cacheKey), "error", saveErr)
							addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheWrite, saveErr), logger)
						}
					} else {
						logger.Warn("Failed to gob-encode cache entry", "error", entryEncodeErr)
						addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, entryEncodeErr), logger)
					}
				} else {
					logger.Warn("Failed to gob-encode analysis data", "error", encodeErr)
					addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheEncode, encodeErr), logger)
				}
			} else {
				logger.Warn("Failed to calculate input hashes for cache save", "error", hashErr)
				addAnalysisError(info, fmt.Errorf("%w: %w", ErrCacheHash, hashErr), logger)
			}
		} else if a.db != nil {
			// Log why saving was skipped
			logger.Debug("Skipping bbolt cache save", "key", string(cacheKey), "load_errors", len(loadErrors), "preamble_empty", info.PromptPreamble == "")
		}
	} // End if !cacheHit

	// Final logging based on outcome
	if cacheHit { // This condition might be less relevant now due to forced re-analysis
		logger.Info("Context analysis finished (cache hit for preamble, re-analyzed for diagnostics)", "decode_duration", loadDuration)
	} else {
		logger.Info("Context analysis finished (full analysis)", "load_duration", loadDuration, "steps_duration", stepsDuration, "preamble_duration", preambleDuration)
	}
	logger.Debug("Final Context Preamble generated", "length", len(info.PromptPreamble))
	logger.Debug("Final Diagnostics collected", "count", len(info.Diagnostics))

	// Return the populated info struct and any fatal error encountered
	return info, analysisErr
}

// InvalidateCache removes the bbolt cached entry for a given directory, typically
// called when go.mod changes.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	logger := stdslog.Default().With("dir", dir)
	a.mu.Lock() // Protect access to db handle
	db := a.db
	a.mu.Unlock()

	if db == nil {
		logger.Debug("Bbolt cache invalidation skipped: DB is nil.")
		return nil // Not an error if caching is disabled
	}
	// Calculate the key based on the directory and current go.mod hash
	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)
	logger.Info("Invalidating bbolt cache entry", "key", string(cacheKey))
	// Use helper function to delete the entry
	return deleteCacheEntryByKey(db, cacheKey, logger) // from deepcomplete_utils.go
}

// InvalidateMemoryCacheForURI clears relevant entries from the ristretto memory cache,
// typically called when a file changes in the editor.
// Currently clears the entire cache for simplicity.
func (a *GoPackagesAnalyzer) InvalidateMemoryCacheForURI(uri string, version int) error {
	logger := stdslog.Default().With("uri", uri, "version", version)
	a.mu.Lock() // Protect access to memoryCache handle
	memCache := a.memoryCache
	a.mu.Unlock()

	if memCache == nil {
		logger.Debug("Memory cache invalidation skipped: Cache is nil.")
		return nil // Not an error if caching is disabled
	}
	// TODO: Implement more granular invalidation if possible/needed.
	// For now, clear the entire cache on any document change.
	logger.Warn("Clearing entire Ristretto memory cache due to document change.", "uri", uri)
	memCache.Clear()
	// Wait for Clear operation to propagate (optional, depends on Ristretto guarantees)
	// memCache.Wait()
	return nil
}

// MemoryCacheEnabled returns true if the Ristretto cache is initialized and available.
func (a *GoPackagesAnalyzer) MemoryCacheEnabled() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.memoryCache != nil
}

// GetMemoryCacheMetrics returns the performance metrics collected by Ristretto.
func (a *GoPackagesAnalyzer) GetMemoryCacheMetrics() *ristretto.Metrics {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.memoryCache != nil {
		return a.memoryCache.Metrics
	}
	return nil // Return nil if cache is not enabled
}

// --- Default Prompt Formatter ---

// templateFormatter implements the PromptFormatter interface using standard Go templates.
type templateFormatter struct{}

// newTemplateFormatter creates a new instance of the default formatter.
func newTemplateFormatter() *templateFormatter { return &templateFormatter{} }

// FormatPrompt combines the context preamble and code snippet into the final LLM prompt string,
// applying truncation limits defined in the configuration.
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate // Use standard or FIM template based on config
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2 // Divide snippet length for FIM prefix/suffix
	logger := stdslog.Default()        // Use default logger

	// Truncate context preamble if it exceeds the limit
	if len(contextPreamble) > maxPreambleLen {
		logger.Warn("Truncating context preamble", "original_length", len(contextPreamble), "max_length", maxPreambleLen)
		// Truncate from the beginning, keeping the end
		marker := "... (context truncated)\n"
		startByte := len(contextPreamble) - maxPreambleLen + len(marker)
		if startByte < 0 {
			startByte = 0 // Avoid negative index
		}
		contextPreamble = marker + contextPreamble[startByte:]
	}

	// Format based on whether FIM is enabled
	if config.UseFim {
		template = config.FimTemplate // Use FIM template
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix

		// Truncate FIM prefix if needed (keep end)
		if len(prefix) > maxFIMPartLen {
			logger.Warn("Truncating FIM prefix", "original_length", len(prefix), "max_length", maxFIMPartLen)
			marker := "...(prefix truncated)"
			startByte := len(prefix) - maxFIMPartLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			prefix = marker + prefix[startByte:]
		}
		// Truncate FIM suffix if needed (keep beginning)
		if len(suffix) > maxFIMPartLen {
			logger.Warn("Truncating FIM suffix", "original_length", len(suffix), "max_length", maxFIMPartLen)
			marker := "(suffix truncated)..."
			endByte := maxFIMPartLen - len(marker)
			if endByte < 0 {
				endByte = 0
			}
			suffix = suffix[:endByte] + marker
		}
		// Format the FIM prompt
		finalPrompt = fmt.Sprintf(template, contextPreamble, prefix, suffix)
	} else {
		// Use standard completion template
		snippet := snippetCtx.Prefix // Use only the prefix for standard completion
		// Truncate snippet (prefix) if needed (keep end)
		if len(snippet) > maxSnippetLen {
			logger.Warn("Truncating code snippet (prefix)", "original_length", len(snippet), "max_length", maxSnippetLen)
			marker := "...(code truncated)\n"
			startByte := len(snippet) - maxSnippetLen + len(marker)
			if startByte < 0 {
				startByte = 0
			}
			snippet = marker + snippet[startByte:]
		}
		// Format the standard prompt
		finalPrompt = fmt.Sprintf(template, contextPreamble, snippet)
	}
	return finalPrompt
}

// =============================================================================
// DeepCompleter Service
// =============================================================================

// DeepCompleter orchestrates the code analysis, prompt formatting, and LLM interaction.
type DeepCompleter struct {
	client    LLMClient       // Interface for interacting with the LLM backend.
	analyzer  Analyzer        // Interface for performing code analysis.
	formatter PromptFormatter // Interface for formatting the LLM prompt.
	config    Config          // Current active configuration.
	configMu  sync.RWMutex    // Mutex to protect concurrent access to config.
}

// NewDeepCompleter creates a new DeepCompleter service instance.
// It loads the configuration using LoadConfig and initializes default components.
// Returns the completer and a potential non-fatal ErrConfig if loading had issues.
func NewDeepCompleter() (*DeepCompleter, error) {
	logger := stdslog.Default() // Use default logger

	// Load configuration (handles defaults, merging, validation, writing defaults)
	cfg, configErr := LoadConfig()
	// Only return fatal errors immediately. Config warnings (ErrConfig) are handled later.
	if configErr != nil && !errors.Is(configErr, ErrConfig) {
		logger.Error("Fatal error during initial config load", "error", configErr)
		return nil, configErr
	}
	// Even with ErrConfig, LoadConfig should return a valid (possibly default) config.
	// Validate again just in case LoadConfig logic changes.
	if err := cfg.Validate(); err != nil {
		logger.Error("Loaded/default config is invalid", "error", err)
		return nil, fmt.Errorf("initial config validation failed: %w", err)
	}

	// Initialize components
	analyzer := NewGoPackagesAnalyzer() // Initialize analyzer (handles its own cache setup)
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(), // Initialize default HTTP Ollama client
		analyzer:  analyzer,
		formatter: newTemplateFormatter(), // Initialize default prompt formatter
		config:    cfg,                    // Set the loaded/validated config
	}

	// Return the completer and the non-fatal config error, if any
	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with a specific,
// provided configuration, bypassing the standard loading process.
// Validates the provided config before creating the completer.
func NewDeepCompleterWithConfig(config Config) (*DeepCompleter, error) {
	// Ensure internal templates are set if missing in provided config
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	// Validate the provided configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	// Initialize components with the provided config
	analyzer := NewGoPackagesAnalyzer()
	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    config,
	}, nil
}

// Close cleans up resources used by the DeepCompleter, primarily the analyzer's caches.
func (dc *DeepCompleter) Close() error {
	if dc.analyzer != nil {
		return dc.analyzer.Close()
	}
	return nil
}

// UpdateConfig atomically updates the completer's configuration after validating the new config.
func (dc *DeepCompleter) UpdateConfig(newConfig Config) error {
	logger := stdslog.Default() // Use default logger
	// Ensure internal templates are set
	if newConfig.PromptTemplate == "" {
		newConfig.PromptTemplate = promptTemplate
	}
	if newConfig.FimTemplate == "" {
		newConfig.FimTemplate = fimPromptTemplate
	}
	// Validate the incoming configuration
	if err := newConfig.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	// Acquire write lock to update config safely
	dc.configMu.Lock()
	defer dc.configMu.Unlock()
	dc.config = newConfig // Update the config atomically

	// Log the updated configuration values
	logger.Info("DeepCompleter configuration updated",
		stdslog.String("ollama_url", newConfig.OllamaURL),
		stdslog.String("model", newConfig.Model),
		stdslog.Int("max_tokens", newConfig.MaxTokens),
		stdslog.Any("stop", newConfig.Stop), // Use Any for slice logging
		stdslog.Float64("temperature", newConfig.Temperature),
		stdslog.String("log_level", newConfig.LogLevel),
		stdslog.Bool("use_ast", newConfig.UseAst),
		stdslog.Bool("use_fim", newConfig.UseFim),
		stdslog.Int("max_preamble_len", newConfig.MaxPreambleLen),
		stdslog.Int("max_snippet_len", newConfig.MaxSnippetLen),
	)
	// TODO: Potentially trigger actions based on config changes (e.g., update logger level)
	return nil
}

// GetCurrentConfig returns a thread-safe copy of the current configuration.
func (dc *DeepCompleter) GetCurrentConfig() Config {
	dc.configMu.RLock() // Acquire read lock
	defer dc.configMu.RUnlock()
	// Create a copy to avoid returning a pointer to the internal config struct
	cfgCopy := dc.config
	// Ensure slices are copied to prevent modification of the internal config via the copy
	if cfgCopy.Stop != nil {
		stopsCopy := make([]string, len(cfgCopy.Stop))
		copy(stopsCopy, cfgCopy.Stop)
		cfgCopy.Stop = stopsCopy
	}
	return cfgCopy
}

// InvalidateAnalyzerCache provides external access to invalidate the analyzer's disk cache.
func (dc *DeepCompleter) InvalidateAnalyzerCache(dir string) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	return dc.analyzer.InvalidateCache(dir)
}

// InvalidateMemoryCacheForURI provides external access to invalidate the analyzer's memory cache.
func (dc *DeepCompleter) InvalidateMemoryCacheForURI(uri string, version int) error {
	if dc.analyzer == nil {
		return errors.New("analyzer not initialized")
	}
	return dc.analyzer.InvalidateMemoryCacheForURI(uri, version)
}

// GetAnalyzer returns the analyzer instance, allowing access to its methods if needed externally.
// Note: Use with caution, prefer dedicated methods on DeepCompleter if possible.
func (dc *DeepCompleter) GetAnalyzer() Analyzer {
	return dc.analyzer
}

// GetCompletion provides basic code completion for a given snippet without file context analysis.
// Primarily useful for the CLI's stdin mode or simple testing.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	logger := stdslog.Default().With("operation", "GetCompletion")
	logger.Info("Handling basic completion request")
	currentConfig := dc.GetCurrentConfig() // Get thread-safe copy of config

	// Use a minimal context preamble for basic completion
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet} // Only prefix is relevant here

	// Format the prompt using the formatter
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	logger.Debug("Generated basic prompt", "length", len(prompt))

	var buffer bytes.Buffer // Buffer to store the streamed response

	// Define the operation to be retried
	apiCallFunc := func() error {
		// Check context before making the call
		select {
		case <-ctx.Done():
			return ctx.Err() // Return context error immediately
		default:
		}

		// Create a context with timeout for the API call itself
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second) // Timeout for API call + streaming
		defer cancelApi()

		logger.Debug("Calling Ollama GenerateStream for basic completion")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig)
		if apiErr != nil {
			// Let retry handler classify the error (unavailable, context, etc.)
			return apiErr
		}

		// Process the stream
		// Use a separate timeout for stream processing within the API call timeout
		streamCtx, cancelStream := context.WithTimeout(apiCtx, 50*time.Second)
		defer cancelStream()
		buffer.Reset()                                            // Clear buffer before processing stream
		streamErr := streamCompletion(streamCtx, reader, &buffer) // Helper from utils
		if streamErr != nil {
			// Wrap stream error for clarity
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		return nil // Success
	}

	// Execute the operation with retry logic
	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // Helper from utils
	if err != nil {
		// Check context cancellation after retries
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
		// Handle specific error types after retries failed
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) {
			logger.Error("Ollama unavailable for basic completion after retries", "error", err)
			return "", fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error for basic completion after retries", "error", err)
			return "", err // Return the specific stream error
		}
		// Handle generic retry failure
		logger.Error("Failed to get basic completion after retries", "error", err)
		return "", fmt.Errorf("failed to get basic completion after %d retries: %w", maxRetries, err)
	}

	logger.Info("Basic completion successful")
	return strings.TrimSpace(buffer.String()), nil // Return the collected completion
}

// GetCompletionStreamFromFile provides context-aware code completion by analyzing the
// specified file and position, then streams the LLM response to the provided writer.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, absFilename string, version int, line, col int, w io.Writer) error {
	logger := stdslog.Default().With("operation", "GetCompletionStreamFromFile", "path", absFilename, "version", version, "line", line, "col", col)
	currentConfig := dc.GetCurrentConfig()                     // Get thread-safe config copy
	var contextPreamble string = "// Basic file context only." // Default preamble if analysis fails/disabled
	var analysisInfo *AstContextInfo
	var analysisErr error

	// Perform code analysis if enabled in config
	if currentConfig.UseAst {
		logger.Info("Analyzing context (or checking cache)")
		// Create a timeout context specifically for the analysis phase
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second) // Shorter timeout for analysis
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, absFilename, version, line, col)
		cancelAnalysis() // Cancel analysis context once done

		// Handle analysis errors
		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			// Fatal analysis error (e.g., panic, critical cache issue)
			logger.Error("Fatal error during analysis/cache check", "error", analysisErr)
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		if analysisErr != nil {
			// Non-fatal analysis error (e.g., type checking errors, minor cache issues)
			logger.Warn("Non-fatal error during analysis/cache check", "error", analysisErr)
			// Append warning to preamble if analysis had issues but produced some context
			if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
				contextPreamble = analysisInfo.PromptPreamble + fmt.Sprintf("\n// Warning: Context analysis completed with errors: %v\n", analysisErr)
			} else {
				contextPreamble += fmt.Sprintf("\n// Warning: Context analysis completed with errors: %v\n", analysisErr)
			}
		} else if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			// Analysis successful and preamble generated
			contextPreamble = analysisInfo.PromptPreamble
		} else {
			// Analysis successful but no preamble generated (or analysisInfo was nil)
			contextPreamble += "\n// Warning: Context analysis returned no specific context preamble.\n"
		}
	} else {
		logger.Info("AST analysis disabled by config.")
	}

	// Extract the code snippet around the cursor
	snippetCtx, snippetErr := extractSnippetContext(absFilename, line, col) // Helper from utils
	if snippetErr != nil {
		logger.Error("Failed to extract code snippet context", "error", snippetErr)
		return fmt.Errorf("failed to extract code snippet context: %w", snippetErr)
	}

	// Format the final prompt
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, currentConfig)
	logger.Debug("Generated prompt", "length", len(prompt))

	// Define the LLM call operation for retry
	apiCallFunc := func() error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		// Context with timeout for the API call + streaming
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()

		logger.Debug("Calling Ollama GenerateStream")
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, currentConfig)
		if apiErr != nil {
			return apiErr // Let retry handler classify
		}
		// Stream the response directly to the provided writer (w)
		streamErr := streamCompletion(apiCtx, reader, w) // Helper from utils
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		return nil // Success
	}

	// Execute LLM call with retry logic
	err := retry(ctx, apiCallFunc, maxRetries, retryDelay, logger) // Helper from utils
	if err != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		// Handle specific errors after retries
		if errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) {
			logger.Error("Ollama unavailable for stream after retries", "error", err)
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		if errors.Is(err, ErrStreamProcessing) {
			logger.Error("Stream processing error for stream after retries", "error", err)
			return err // Return specific stream error
		}
		// Handle generic retry failure
		logger.Error("Failed to get completion stream after retries", "error", err)
		return fmt.Errorf("failed to get completion stream after %d retries: %w", maxRetries, err)
	}

	// Log success, potentially noting analysis warnings
	if analysisErr != nil {
		logger.Warn("Completion stream successful, but context analysis encountered non-fatal errors", "analysis_error", analysisErr)
	} else {
		logger.Info("Completion stream successful")
	}
	return nil // Return nil on success
}
