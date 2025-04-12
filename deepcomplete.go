// deepcomplete/deepcomplete.go
package deepcomplete

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors" // Using errors.Join requires Go 1.20+
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug" // For panic recovery stack trace
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/ast/astutil" // Utility for AST traversal
	"golang.org/x/tools/go/packages"    // Standard way to load packages for analysis
)

// =============================================================================
// Constants & Core Types
// =============================================================================

const (
	defaultOllamaURL = "http://localhost:11434"
	defaultModel     = "deepseek-coder-r2"
	// Standard prompt template - Refined system message
	promptTemplate = `<s>[INST] <<SYS>>
You are an expert Go programming assistant.
Analyze the provided context (enclosing function, imports, scope variables, comments, code structure) and the preceding code snippet.
Complete the Go code accurately and concisely.
Output ONLY the raw Go code completion, without any markdown, explanations, or introductory text.
<</SYS>>

CONTEXT:
%s

CODE SNIPPET TO COMPLETE:
` + "```go\n%s\n```" + `
[/INST]`
	// FIM prompt template - Refined system message (generic example)
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

	maxRetries       = 3
	retryDelay       = 2 * time.Second
	maxContentLength = 4096 // Max length for combined prefix/suffix in FIM, or prefix in standard
	defaultMaxTokens = 256
	// DefaultStop is exported for potential use in CLI default flags
	DefaultStop        = "\n" // Common stop sequences for code completion
	defaultTemperature = 0.1
	// Default config file name
	defaultConfigFileName = "config.json"
	configDirName         = "deepcomplete" // Subdirectory name for config/data
)

// Config holds the active configuration for the autocompletion. Exported.
type Config struct {
	OllamaURL      string   `json:"ollama_url"`
	Model          string   `json:"model"`
	PromptTemplate string   `json:"-"` // Loaded internally, not from file
	FimTemplate    string   `json:"-"` // Loaded internally, not from file
	MaxTokens      int      `json:"max_tokens"`
	Stop           []string `json:"stop"`
	Temperature    float64  `json:"temperature"`
	UseAst         bool     `json:"use_ast"` // Use AST and Type analysis (preferred)
	UseFim         bool     `json:"use_fim"` // Use Fill-in-the-Middle prompting
}

// FileConfig represents the structure of the JSON config file.
// Uses pointers to distinguish between zero values and unset fields during merge.
type FileConfig struct {
	OllamaURL   *string   `json:"ollama_url"`
	Model       *string   `json:"model"`
	MaxTokens   *int      `json:"max_tokens"`
	Stop        *[]string `json:"stop"`
	Temperature *float64  `json:"temperature"`
	UseAst      *bool     `json:"use_ast"`
	UseFim      *bool     `json:"use_fim"`
}

// AstContextInfo holds structured information extracted from AST/Types analysis. Exported.
type AstContextInfo struct {
	FilePath           string
	CursorPos          token.Pos
	PackageName        string
	EnclosingFunc      *types.Func             // Function or method containing the cursor
	EnclosingFuncNode  *ast.FuncDecl           // AST node for the function
	EnclosingBlock     *ast.BlockStmt          // Innermost block statement
	Imports            []*ast.ImportSpec       // Populated with file imports
	CommentsNearCursor []string                // Comments found near the cursor
	IdentifierAtCursor *ast.Ident              // Identifier immediately before/at cursor
	IdentifierType     types.Type              // Type of the identifier at cursor (if resolved)
	IdentifierObject   types.Object            // Object the identifier resolves to
	SelectorExpr       *ast.SelectorExpr       // Selector expression active at cursor (e.g., x.y|)
	SelectorExprType   types.Type              // Type of the expression being selected from (e.g., type of x in x.y|)
	CallExpr           *ast.CallExpr           // Call expression active at cursor (e.g., myFunc(|))
	CallExprFuncType   *types.Signature        // Type signature of the function being called
	CallArgIndex       int                     // Index of the argument the cursor is likely in (0-based)
	VariablesInScope   map[string]types.Object // Map variable name to its type object
	PromptPreamble     string                  // Formatted context for the LLM prompt
	AnalysisErrors     []error                 // Non-fatal errors encountered during analysis
}

// SnippetContext holds the code prefix and suffix for prompting.
type SnippetContext struct {
	Prefix   string
	Suffix   string
	FullLine string // The full line where the cursor is located
}

// OllamaError defines a custom error for Ollama API issues (internal but used by retry).
type OllamaError struct {
	Message string
	Status  int // HTTP status code
}

func (e *OllamaError) Error() string {
	return fmt.Sprintf("Ollama error: %s (Status: %d)", e.Message, e.Status)
}

// OllamaResponse represents the streaming response structure (internal).
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"` // Check this field in the stream
}

// =============================================================================
// Interfaces for Components
// =============================================================================

// LLMClient defines the interface for interacting with the LLM API.
type LLMClient interface {
	GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error)
}

// Analyzer defines the interface for analyzing code context. Exported for testability.
type Analyzer interface {
	Analyze(ctx context.Context, filename string, line, col int) (*AstContextInfo, error)
}

// PromptFormatter defines the interface for formatting the final prompt.
type PromptFormatter interface {
	FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string
}

// =============================================================================
// Variables & Default Config
// =============================================================================

var (
	// DefaultConfig enables AST context by default. Exported.
	DefaultConfig = Config{
		OllamaURL:      defaultOllamaURL,
		Model:          defaultModel,
		PromptTemplate: promptTemplate,    // Set default template
		FimTemplate:    fimPromptTemplate, // Set default FIM template
		MaxTokens:      defaultMaxTokens,
		Stop:           []string{DefaultStop, "}", "//", "/*"}, // Use exported DefaultStop
		Temperature:    defaultTemperature,
		UseAst:         true,  // Enable AST context by default
		UseFim:         false, // Disable FIM by default
	}

	// Pastel color codes for terminal output. Exported.
	ColorReset  = "\033[0m"
	ColorGreen  = "\033[38;5;119m"
	ColorYellow = "\033[38;5;220m"
	ColorBlue   = "\033[38;5;153m"
	ColorRed    = "\033[38;5;203m"
	ColorCyan   = "\033[38;5;141m"
)

// =============================================================================
// Exported Helper Functions (e.g., for CLI)
// =============================================================================

// PrettyPrint prints colored text to the terminal. Exported for CLI use.
func PrettyPrint(color, text string) {
	fmt.Print(color, text, ColorReset)
}

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadConfig loads configuration from standard locations, merges with defaults,
// and attempts to write a default config file if none exists.
// Returns the loaded config and any non-fatal errors encountered during loading/parsing/writing.
func LoadConfig() (Config, error) {
	cfg := DefaultConfig // Start with defaults
	var loadedFromFile bool
	var loadErrors []error

	// Determine primary and secondary config paths
	primaryPath, secondaryPath, pathErr := getConfigPaths()
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		log.Printf("Warning: Could not determine config paths: %v", pathErr)
	}

	// Try loading from primary path (e.g., XDG_CONFIG_HOME)
	if primaryPath != "" {
		loaded, loadErr := loadAndMergeConfig(primaryPath, &cfg)
		if loadErr != nil {
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", primaryPath, loadErr))
		}
		loadedFromFile = loaded
		if loaded {
			log.Printf("Loaded config from %s", primaryPath)
		}
	}

	// Try loading from secondary path if not loaded from primary
	if !loadedFromFile && secondaryPath != "" {
		loaded, loadErr := loadAndMergeConfig(secondaryPath, &cfg)
		if loadErr != nil {
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", secondaryPath, loadErr))
		}
		loadedFromFile = loaded
		if loaded {
			log.Printf("Loaded config from %s", secondaryPath)
		}
	}

	// If no config file was found/loaded successfully, try to write the default one
	if !loadedFromFile {
		log.Println("No valid config file found in standard locations.")
		writePath := primaryPath // Prefer primary path for writing
		if writePath == "" {
			writePath = secondaryPath
		} // Fallback if primary failed

		if writePath != "" {
			log.Printf("Attempting to write default config to %s", writePath)
			if err := writeDefaultConfig(writePath, DefaultConfig); err != nil {
				log.Printf("Warning: Failed to write default config: %v", err)
				loadErrors = append(loadErrors, fmt.Errorf("writing default config failed: %w", err))
			}
		} else {
			log.Println("Warning: Cannot determine path to write default config.")
			loadErrors = append(loadErrors, errors.New("cannot determine default config path"))
		}
	}

	// Ensure essential internal templates are set if missing
	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	return cfg, errors.Join(loadErrors...) // Return loaded/default config and combined non-fatal errors
}

// getConfigPaths determines the primary (XDG) and secondary (~/.local/share) config paths.
func getConfigPaths() (primary string, secondary string, err error) {
	var cfgErr, homeErr error

	userConfigDir, cfgErr := os.UserConfigDir()
	if cfgErr == nil {
		primary = filepath.Join(userConfigDir, configDirName, defaultConfigFileName)
	} else {
		log.Printf("Warning: Could not determine user config directory: %v", cfgErr)
	}

	homeDir, homeErr := os.UserHomeDir()
	if homeErr == nil {
		secondary = filepath.Join(homeDir, ".local", "share", configDirName, defaultConfigFileName)
	} else {
		log.Printf("Warning: Could not determine user home directory: %v", homeErr)
	}

	// Return combined error if both failed
	if cfgErr != nil && homeErr != nil {
		err = fmt.Errorf("cannot determine config/home directories: %w; %w", cfgErr, homeErr)
	}

	return primary, secondary, err
}

// loadAndMergeConfig attempts to load config from a path and merge it into cfg.
func loadAndMergeConfig(path string, cfg *Config) (bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return true, fmt.Errorf("reading config file failed: %w", err) // File exists but read failed
	}
	if len(data) == 0 { // Handle empty file case
		log.Printf("Warning: Config file exists but is empty: %s", path)
		return true, nil // Treat as loaded but with no overrides
	}

	var fileCfg FileConfig
	if err := json.Unmarshal(data, &fileCfg); err != nil {
		return true, fmt.Errorf("parsing config file JSON failed: %w", err) // Invalid JSON
	}

	// Merge non-nil values from fileCfg into cfg
	if fileCfg.OllamaURL != nil {
		cfg.OllamaURL = *fileCfg.OllamaURL
	}
	if fileCfg.Model != nil {
		cfg.Model = *fileCfg.Model
	}
	if fileCfg.MaxTokens != nil {
		cfg.MaxTokens = *fileCfg.MaxTokens
	}
	if fileCfg.Stop != nil {
		cfg.Stop = *fileCfg.Stop
	}
	if fileCfg.Temperature != nil {
		cfg.Temperature = *fileCfg.Temperature
	}
	if fileCfg.UseAst != nil {
		cfg.UseAst = *fileCfg.UseAst
	}
	if fileCfg.UseFim != nil {
		cfg.UseFim = *fileCfg.UseFim
	}

	return true, nil
}

// writeDefaultConfig creates the directory and writes the default config as JSON.
func writeDefaultConfig(path string, defaultConfig Config) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0750); err != nil { // Use 0750 for permissions
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}

	// Marshal the *actual* Config struct (not FileConfig) for defaults
	// Need to make fields exportable for MarshalIndent to work correctly
	// Let's marshal a temporary struct with exported fields matching JSON tags
	type ExportableConfig struct {
		OllamaURL   string   `json:"ollama_url"`
		Model       string   `json:"model"`
		MaxTokens   int      `json:"max_tokens"`
		Stop        []string `json:"stop"`
		Temperature float64  `json:"temperature"`
		UseAst      bool     `json:"use_ast"`
		UseFim      bool     `json:"use_fim"`
	}
	expCfg := ExportableConfig{
		OllamaURL:   defaultConfig.OllamaURL,
		Model:       defaultConfig.Model,
		MaxTokens:   defaultConfig.MaxTokens,
		Stop:        defaultConfig.Stop,
		Temperature: defaultConfig.Temperature,
		UseAst:      defaultConfig.UseAst,
		UseFim:      defaultConfig.UseFim,
	}

	jsonData, err := json.MarshalIndent(expCfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal default config to JSON: %w", err)
	}

	if err := os.WriteFile(path, jsonData, 0640); err != nil { // Use 0640 for permissions
		return fmt.Errorf("failed to write default config file %s: %w", path, err)
	}
	log.Printf("Wrote default configuration to %s", path)
	return nil
}

// =============================================================================
// Default Implementations (Continued)
// =============================================================================

// --- Ollama Client ---

// httpOllamaClient implements LLMClient using standard HTTP calls.
type httpOllamaClient struct {
	httpClient *http.Client
}

// newHttpOllamaClient creates a new client for Ollama interaction. (unexported)
func newHttpOllamaClient() *httpOllamaClient {
	return &httpOllamaClient{
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
}

// GenerateStream handles the HTTP request to the Ollama generate endpoint.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	u, err := url.Parse(config.OllamaURL + "/api/generate")
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL: %w", err)
	}
	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"temperature": config.Temperature,
			"num_ctx":     4096,
			"top_p":       0.9,
			"stop":        config.Stop,
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
	req.Header.Set("Accept", "application/x-ndjson")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("ollama request timed out: %w", err)
		}
		return nil, fmt.Errorf("error making HTTP request to Ollama: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(could not read error body)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error
			}
		}
		if resp.StatusCode == http.StatusNotFound {
			return nil, &OllamaError{Message: fmt.Sprintf("Ollama API endpoint not found or model '%s' not available: %s", config.Model, bodyString), Status: resp.StatusCode}
		}
		if resp.StatusCode == http.StatusServiceUnavailable || resp.StatusCode == http.StatusTooManyRequests {
			return nil, &OllamaError{Message: fmt.Sprintf("Ollama service unavailable or busy: %s", bodyString), Status: resp.StatusCode}
		}
		return nil, &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
	}
	return resp.Body, nil
}

// --- Code Analyzer ---

// GoPackagesAnalyzer implements Analyzer using go/packages. Exported for testability.
type GoPackagesAnalyzer struct{}

// NewGoPackagesAnalyzer creates a new Go code analyzer. Exported for testability.
func NewGoPackagesAnalyzer() *GoPackagesAnalyzer {
	return &GoPackagesAnalyzer{}
}

// Analyze parses the file, performs type checking, and extracts context.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, filename string, line, col int) (info *AstContextInfo, analysisErr error) {
	// Initialize context info struct
	info = &AstContextInfo{
		FilePath:         filename,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		CallArgIndex:     -1,
	}

	// Add panic recovery
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Panic recovered during AnalyzeCodeContext: %v\n%s", r, string(debug.Stack()))
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			addAnalysisError(info, panicErr) // Use helper
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr)
			}
		}
	}()

	absFilename, err := filepath.Abs(filename)
	if err != nil {
		return info, fmt.Errorf("failed to get absolute path for '%s': %w", filename, err)
	}
	info.FilePath = absFilename
	log.Printf("Starting context analysis for: %s (%d:%d)", absFilename, line, col)

	// --- Load Package & File Info ---
	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadErrors := loadPackageInfo(ctx, absFilename, fset)
	for _, loadErr := range loadErrors {
		addAnalysisError(info, loadErr)
	} // Use helper

	// --- Fallback Parsing if loading failed ---
	if targetFileAST == nil {
		log.Printf("Attempting direct parse of %s", absFilename)
		targetFileAST, targetFile, err = directParse(absFilename, fset)
		if err != nil {
			addAnalysisError(info, err)
			analysisErr = errors.Join(info.AnalysisErrors...)
			return info, analysisErr
		} // Cannot proceed
		if targetFileAST.Name != nil {
			info.PackageName = targetFileAST.Name.Name
		} else {
			info.PackageName = "main"
		}
		log.Printf("Using AST from direct parse. Package: %s", info.PackageName)
		targetPkg = nil // No type info
	} else if targetPkg != nil {
		if targetPkg.TypesInfo == nil {
			log.Println("Warning: targetPkg found but TypesInfo is nil.")
			addAnalysisError(info, errors.New("package type info missing"))
		}
		info.PackageName = targetPkg.Name
	} else {
		if targetFileAST.Name != nil {
			info.PackageName = targetFileAST.Name.Name
		} else {
			info.PackageName = "main"
		}
	}

	// Populate Imports
	if targetFileAST != nil {
		info.Imports = targetFileAST.Imports
	}

	// --- Perform Analysis Steps ---
	err = a.performAnalysisSteps(targetFile, targetFileAST, targetPkg, fset, line, col, info) // Pass info
	if err != nil {
		addAnalysisError(info, err)
	} // Add error from analysis steps

	// --- Build Preamble ---
	info.PromptPreamble = buildPreamble(info, targetPkg) // Pass info
	log.Printf("Generated Context Preamble:\n---\n%s\n---", info.PromptPreamble)
	logAnalysisErrors(info.AnalysisErrors) // Log all collected non-fatal errors

	analysisErr = errors.Join(info.AnalysisErrors...)
	return info, analysisErr
}

// performAnalysisSteps encapsulates the core analysis logic after loading/parsing. (unexported helper)
func (a *GoPackagesAnalyzer) performAnalysisSteps(
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package,
	fset *token.FileSet,
	line, col int,
	info *AstContextInfo, // Modifies info directly
) error { // Returns potential fatal error during steps

	// --- Calculate Cursor Position ---
	if targetFile == nil {
		return errors.New("failed to get token.File for position calculation")
	}
	cursorPos, posErr := calculateCursorPos(targetFile, line, col)
	if posErr != nil {
		return fmt.Errorf("cursor position error: %w", posErr)
	} // Return fatal error
	info.CursorPos = cursorPos
	log.Printf("Calculated cursor token.Pos: %d (Line: %d, Col: %d)", info.CursorPos, fset.Position(info.CursorPos).Line, fset.Position(info.CursorPos).Column)

	// --- Find Path & Context Nodes ---
	path := findEnclosingPath(targetFileAST, info.CursorPos, info) // Pass info to store non-fatal errors
	if path != nil {
		findContextNodes(path, info.CursorPos, targetPkg, info) // Modifies info, adds non-fatal errors
	}

	// --- Gather Scope Context ---
	gatherScopeContext(path, targetPkg, info) // Modifies info, adds non-fatal errors

	// --- Find Comments ---
	findRelevantComments(targetFileAST, path, info.CursorPos, fset, info) // Modifies info

	return nil // Return nil, non-fatal errors are in info.AnalysisErrors
}

// --- Prompt Formatter ---

// templateFormatter implements PromptFormatter using standard templates.
type templateFormatter struct{}

// newTemplateFormatter creates a new prompt formatter. (unexported)
func newTemplateFormatter() *templateFormatter {
	return &templateFormatter{}
}

// FormatPrompt generates the final prompt string.
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate
	snippet := snippetCtx.Prefix // Default to prefix for standard

	if config.UseFim {
		template = config.FimTemplate
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix
		// Basic truncation
		combinedLen := len(prefix) + len(suffix)
		if combinedLen > maxContentLength {
			allowedLen := maxContentLength / 2
			if len(prefix) > allowedLen {
				prefix = "..." + prefix[len(prefix)-allowedLen+3:]
			}
			if len(suffix) > allowedLen {
				suffix = suffix[:allowedLen-3] + "..."
			}
			log.Printf("Truncated FIM prefix/suffix.")
		}
		// Use FIM template format (adjust markers if needed for model)
		finalPrompt = fmt.Sprintf(template, prefix, suffix)
		// Prepend context preamble (model might ignore if not trained for it)
		finalPrompt = contextPreamble + "\n" + finalPrompt
	} else {
		// Standard Prompt
		if len(snippet) > maxContentLength {
			snippet = "..." + snippet[len(snippet)-maxContentLength+3:]
			log.Printf("Truncated code snippet.")
		}
		maxPreambleLen := 1024
		if len(contextPreamble) > maxPreambleLen {
			contextPreamble = contextPreamble[:maxPreambleLen] + "\n... (context truncated)"
			log.Printf("Truncated context preamble.")
		}
		finalPrompt = fmt.Sprintf(template, contextPreamble, snippet)
	}
	return finalPrompt
}

// =============================================================================
// DeepCompleter Service
// =============================================================================

// DeepCompleter orchestrates the code completion process. Exported.
type DeepCompleter struct {
	client    LLMClient
	analyzer  Analyzer
	formatter PromptFormatter
	config    Config
}

// NewDeepCompleter creates a new DeepCompleter service with default components.
// It loads configuration from standard locations and applies defaults. Exported.
func NewDeepCompleter() (*DeepCompleter, error) {
	cfg, err := LoadConfig() // Load defaults + file config
	if err != nil {
		// Log error but potentially continue with defaults
		log.Printf("Warning: Error loading initial config: %v", err)
		// In case of error, ensure cfg is still usable (defaults)
		cfg = DefaultConfig // Use the package-level DefaultConfig var
	}

	return &DeepCompleter{
		client:    newHttpOllamaClient(),   // Use unexported constructor
		analyzer:  NewGoPackagesAnalyzer(), // Use exported constructor
		formatter: newTemplateFormatter(),  // Use unexported constructor
		config:    cfg,
	}, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with provided config
// and default components. Exported.
func NewDeepCompleterWithConfig(config Config) *DeepCompleter {
	// Ensure templates have defaults if not set in provided config
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}

	return &DeepCompleter{
		client:    newHttpOllamaClient(),   // Use unexported constructor
		analyzer:  NewGoPackagesAnalyzer(), // Use exported constructor
		formatter: newTemplateFormatter(),  // Use unexported constructor
		config:    config,
	}
}

// GetCompletion provides completion for a raw code snippet (no file context). Non-streaming. Exported.
// Calls GenerateStream internally and buffers the result.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	log.Println("DeepCompleter.GetCompletion called for basic prompt.")
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, dc.config)

	reader, err := dc.client.GenerateStream(ctx, prompt, dc.config)
	if err != nil {
		return "", fmt.Errorf("failed to call Ollama API for basic prompt: %w", err)
	}

	var buffer bytes.Buffer
	streamCtx, cancelStream := context.WithTimeout(ctx, 50*time.Second)
	defer cancelStream()
	// Use the package-level streamCompletion helper
	if streamErr := streamCompletion(streamCtx, reader, &buffer); streamErr != nil {
		return "", fmt.Errorf("error processing stream for basic prompt: %w", streamErr)
	}
	return buffer.String(), nil
}

// GetCompletionStreamFromFile is the primary function using AST context and streaming output. Exported.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, filename string, row, col int, w io.Writer) error {
	var contextPreamble string = "// Basic file context only.\n"
	var analysisErr error // Stores combined non-fatal errors from analysis

	// --- 1. Analyze Context ---
	if dc.config.UseAst {
		log.Printf("Analyzing context using AST/Types for %s:%d:%d", filename, row, col)
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 20*time.Second)
		var analysisInfo *AstContextInfo
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, filename, row, col) // Use analyzer component
		cancelAnalysis()
		if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else {
			contextPreamble += "// Warning: AST analysis returned no specific context.\n"
		}
		if analysisErr != nil {
			logAnalysisErrors([]error{analysisErr})
		} // Log non-fatal errors
	} else {
		log.Println("AST analysis disabled.")
	}

	// --- 2. Extract Snippet ---
	snippetCtx, err := extractSnippetContext(filename, row, col)
	if err != nil {
		return fmt.Errorf("failed to extract code snippet context: %w", err)
	} // Fatal error

	// --- 3. Format Prompt ---
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, dc.config) // Use formatter component
	log.Printf("Generated Prompt (length %d):\n---\n%s\n---", len(prompt), prompt)

	// --- 4. Call LLM with Retry ---
	apiCallFunc := func() error {
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, dc.config) // Use client component
		if apiErr != nil {
			if analysisErr != nil {
				log.Printf("API error (%v) potentially related to prior analysis errors (%v)", apiErr, analysisErr)
			}
			return apiErr
		} // Let retry handle check
		PrettyPrint(ColorGreen, "Completion:\n") // Use exported PrettyPrint
		// Use the package-level streamCompletion helper
		streamErr := streamCompletion(apiCtx, reader, w)
		fmt.Fprintln(w) // Final newline
		if streamErr != nil {
			if errors.Is(streamErr, context.Canceled) || errors.Is(streamErr, context.DeadlineExceeded) {
				return streamErr
			}
			return fmt.Errorf("error during completion streaming: %w", streamErr)
		}
		log.Println("Completion stream finished successfully for this attempt.")
		return nil
	}

	err = retry(ctx, apiCallFunc, maxRetries, retryDelay) // Use retry helper defined below
	if err != nil {
		return fmt.Errorf("failed to get completion stream after retries: %w", err)
	} // Fatal if retries fail

	// Log analysis errors again as a final warning, even if completion succeeded
	if analysisErr != nil {
		log.Printf("Warning: Completion succeeded, but context analysis encountered errors: %v", analysisErr)
	}
	return nil // Overall success
}

// =============================================================================
// Internal Helpers (Unexported package-level functions)
// =============================================================================

// --- Ollama Stream Processing Helpers ---

// streamCompletion reads and processes the NDJSON stream from Ollama.
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer) error {
	defer r.Close()
	reader := bufio.NewReader(r)
	for {
		select {
		case <-ctx.Done():
			log.Println("Context cancelled during streaming")
			return ctx.Err()
		default:
		}
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if len(line) > 0 {
					if procErr := processLine(line, w); procErr != nil {
						log.Printf("Error processing final line before EOF: %v", procErr)
						return procErr
					}
				}
				return nil
			}
			log.Printf("Error reading from Ollama stream: %v", err)
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			}
		}
		if procErr := processLine(line, w); procErr != nil {
			log.Printf("Error processing Ollama response line: %v", procErr)
			return procErr
		}
	}
}

// processLine unmarshals and handles a single line from the Ollama stream.
func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil
	}
	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		log.Printf("Error unmarshalling Ollama JSON line: %v, line content: '%s'", err, string(line))
		return nil
	}
	if resp.Error != "" {
		log.Printf("Ollama reported an error in stream: %s", resp.Error)
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	}
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		log.Printf("Error writing completion chunk to output: %v", err)
		return fmt.Errorf("error writing to output: %w", err)
	}
	return nil
}

// --- Retry Helper ---

// retry implements a retry mechanism with backoff for specific errors.
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration) error {
	var err error
	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		err = operation()
		if err == nil {
			return nil
		}
		var ollamaErr *OllamaError
		isRetryable := errors.As(err, &ollamaErr) && (ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		if !isRetryable {
			log.Printf("Non-retryable error encountered: %v", err)
			return err
		}
		log.Printf("Attempt %d failed with retryable error: %v. Retrying in %v...", i+1, err, currentDelay)
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled during retry wait: %v", ctx.Err())
			return ctx.Err()
		case <-time.After(currentDelay):
		}
	}
	log.Printf("Operation failed after %d retries.", maxRetries)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, err)
}

// --- Analysis Helpers ---

// loadPackageInfo loads package information using go/packages.
func loadPackageInfo(ctx context.Context, absFilename string, fset *token.FileSet) (*packages.Package, *ast.File, *token.File, []error) {
	dir := filepath.Dir(absFilename)
	var loadErrors []error
	cfg := &packages.Config{Context: ctx, Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedTypes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedTypesSizes | packages.NeedDeps, Dir: dir, Fset: fset, ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		const mode = parser.ParseComments | parser.AllErrors
		file, err := parser.ParseFile(fset, filename, src, mode)
		if err != nil {
			log.Printf("Parser error in %s (ignored for partial AST): %v", filename, err)
		}
		return file, nil
	}, Tests: true}
	pkgs, loadErr := packages.Load(cfg, ".")
	if loadErr != nil {
		log.Printf("Error loading packages: %v", loadErr)
		loadErrors = append(loadErrors, fmt.Errorf("package loading failed: %w", loadErr))
	}
	if packages.PrintErrors(pkgs) > 0 {
		log.Println("Detailed errors encountered during package loading")
		loadErrors = append(loadErrors, errors.New("package loading reported errors"))
	}
	for _, pkg := range pkgs {
		if pkg == nil || pkg.Fset == nil || pkg.Syntax == nil || pkg.CompiledGoFiles == nil {
			continue
		}
		for i, _ := range pkg.CompiledGoFiles { /* FIX: Use _ for filePath */
			if i >= len(pkg.Syntax) {
				continue
			}
			astNode := pkg.Syntax[i]
			if astNode == nil {
				continue
			}
			file := pkg.Fset.File(astNode.Pos())
			if file != nil && file.Name() == absFilename {
				return pkg, astNode, file, loadErrors
			}
		}
	}
	return nil, nil, nil, loadErrors
}

// directParse attempts to parse a single file directly.
func directParse(absFilename string, fset *token.FileSet) (*ast.File, *token.File, error) {
	srcBytes, readErr := os.ReadFile(absFilename)
	if readErr != nil {
		return nil, nil, fmt.Errorf("direct read failed for '%s': %w", absFilename, readErr)
	}
	const mode = parser.ParseComments | parser.AllErrors
	targetFileAST, parseErr := parser.ParseFile(fset, absFilename, srcBytes, mode)
	if parseErr != nil {
		log.Printf("Direct parsing of %s failed: %v", absFilename, parseErr)
	}
	if targetFileAST == nil {
		return nil, nil, errors.New("failed to obtain any AST from direct parse")
	}
	targetFile := fset.File(targetFileAST.Pos())
	return targetFileAST, targetFile, parseErr
}

// findEnclosingPath finds the AST path to the cursor.
func findEnclosingPath(targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo) []ast.Node {
	if targetFileAST == nil {
		addAnalysisError(info, errors.New("cannot find enclosing path: targetFileAST is nil"))
		return nil
	}
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil {
		addAnalysisError(info, errors.New("failed to find AST path enclosing cursor"))
		log.Println("Failed to find AST path enclosing cursor.")
	}
	return path
}

// gatherScopeContext extracts enclosing function/block and populates scope variables.
func gatherScopeContext(path []ast.Node, targetPkg *packages.Package, info *AstContextInfo) {
	if path == nil {
		return
	}
	for i := len(path) - 1; i >= 0; i-- {
		node := path[i]
		switch n := node.(type) {
		case *ast.FuncDecl:
			if info.EnclosingFuncNode == nil {
				info.EnclosingFuncNode = n
			}
			if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
				if obj, ok := targetPkg.TypesInfo.Defs[n.Name]; ok && obj != nil {
					if fn, ok := obj.(*types.Func); ok {
						info.EnclosingFunc = fn
						if sig, ok := fn.Type().(*types.Signature); ok {
							addSignatureToScope(sig, info.VariablesInScope)
						} else {
							log.Printf("Warning: Could not assert *types.Signature for func %s", n.Name.Name)
						}
					}
				}
			}
		case *ast.BlockStmt:
			if info.EnclosingBlock == nil {
				info.EnclosingBlock = n
			}
			if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
				if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
					addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
				}
			} else {
				if targetPkg != nil {
					log.Println("Cannot extract block scope variables: missing type info.")
				}
			}
		}
	}
	addPackageScope(targetPkg, info)
}

// addPackageScope adds package-level declarations to the scope map.
func addPackageScope(targetPkg *packages.Package, info *AstContextInfo) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope)
		} else {
			addAnalysisError(info, errors.New("package scope missing"))
		}
	} else {
		log.Println("Package type info unavailable for package scope.")
		if targetPkg != nil {
			addAnalysisError(info, errors.New("package type info missing"))
		}
	}
}

// findRelevantComments finds comments near the cursor.
func findRelevantComments(targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo) {
	var cmap ast.CommentMap
	if targetFileAST != nil {
		cmap = ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	} else {
		addAnalysisError(info, errors.New("cannot find comments: targetFileAST is nil"))
		return
	}
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, info.CursorPos, fset)
}

// buildPreamble formats the collected context information into a string.
func buildPreamble(info *AstContextInfo, targetPkg *packages.Package) string {
	var preamble strings.Builder
	qualifier := types.RelativeTo(targetPkg.Types)
	preamble.WriteString(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), info.PackageName))
	if len(info.Imports) > 0 {
		preamble.WriteString("// Imports:\n")
		for _, imp := range info.Imports {
			path := ""
			if imp.Path != nil {
				path = imp.Path.Value
			}
			name := ""
			if imp.Name != nil {
				name = imp.Name.Name + " "
			}
			preamble.WriteString(fmt.Sprintf("//   import %s%s\n", name, path))
		}
	}
	if info.EnclosingFunc != nil {
		preamble.WriteString(fmt.Sprintf("// Enclosing Function: %s\n", types.ObjectString(info.EnclosingFunc, qualifier)))
	} else if info.EnclosingFuncNode != nil {
		preamble.WriteString(fmt.Sprintf("// Enclosing Function (AST): %s\n", formatFuncSignature(info.EnclosingFuncNode)))
	}
	if len(info.CommentsNearCursor) > 0 {
		preamble.WriteString("// Relevant Comments:\n")
		for _, c := range info.CommentsNearCursor {
			cleanComment := strings.TrimSpace(strings.TrimPrefix(c, "//"))
			cleanComment = strings.TrimSpace(strings.TrimPrefix(cleanComment, "/*"))
			cleanComment = strings.TrimSpace(strings.TrimSuffix(cleanComment, "*/"))
			preamble.WriteString(fmt.Sprintf("//   %s\n", cleanComment))
		}
	}
	if info.CallExpr != nil {
		funcName := "[unknown function]"
		switch fun := info.CallExpr.Fun.(type) {
		case *ast.Ident:
			funcName = fun.Name
		case *ast.SelectorExpr:
			if fun.Sel != nil {
				funcName = fun.Sel.Name
			}
		}
		preamble.WriteString(fmt.Sprintf("// Inside function call: %s (Arg %d)\n", funcName, info.CallArgIndex+1))
		if info.CallExprFuncType != nil {
			preamble.WriteString(fmt.Sprintf("// Function Signature: %s\n", types.TypeString(info.CallExprFuncType, qualifier)))
		} else {
			preamble.WriteString("// Function Signature: (unknown)\n")
		}
	} else if info.SelectorExpr != nil && info.SelectorExprType != nil {
		selName := ""
		if info.SelectorExpr.Sel != nil {
			selName = info.SelectorExpr.Sel.Name
		}
		typeName := types.TypeString(info.SelectorExprType, qualifier)
		preamble.WriteString(fmt.Sprintf("// Selector context: expr type = %s (selecting '%s')\n", typeName, selName))
		members := listTypeMembers(info.SelectorExprType, targetPkg)
		if len(members) > 0 {
			preamble.WriteString("//   Available members:\n")
			sort.Strings(members)
			for _, member := range members {
				preamble.WriteString(fmt.Sprintf("//     - %s\n", member))
			}
		}
	} else if info.IdentifierAtCursor != nil {
		identName := info.IdentifierAtCursor.Name
		if info.IdentifierType != nil {
			typeName := types.TypeString(info.IdentifierType, qualifier)
			preamble.WriteString(fmt.Sprintf("// Identifier at cursor: %s (Type: %s)\n", identName, typeName))
		} else if info.IdentifierObject != nil {
			preamble.WriteString(fmt.Sprintf("// Identifier at cursor: %s (Object: %s)\n", identName, info.IdentifierObject.Name()))
		} else {
			preamble.WriteString(fmt.Sprintf("// Identifier at cursor: %s (Type unknown)\n", identName))
		}
	}
	if len(info.VariablesInScope) > 0 {
		preamble.WriteString("// Variables/Constants/Types in Scope:\n")
		var names []string
		for name := range info.VariablesInScope {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			obj := info.VariablesInScope[name]
			preamble.WriteString(fmt.Sprintf("//   %s\n", types.ObjectString(obj, qualifier)))
		}
	}
	return preamble.String()
}

// calculateCursorPos converts 1-based line/col to token.Pos (internal).
func calculateCursorPos(file *token.File, line, col int) (token.Pos, error) {
	if line <= 0 {
		return token.NoPos, fmt.Errorf("invalid line number: %d (must be >= 1)", line)
	}
	if line > file.LineCount() {
		return token.NoPos, fmt.Errorf("line number %d exceeds file line count %d", line, file.LineCount())
	}
	lineStartOffset := file.LineStart(line)
	if !lineStartOffset.IsValid() {
		return token.NoPos, fmt.Errorf("cannot get start offset for line %d in file '%s'", line, file.Name())
	}
	cursorOffset := int(lineStartOffset) + col - 1
	maxOffset := file.Size()
	if cursorOffset < int(file.Pos(0)) || cursorOffset > maxOffset {
		log.Printf("Warning: column %d results in offset %d which is outside valid range [0, %d] for line %d. Clamping.", col, cursorOffset, maxOffset, line)
		if cursorOffset > maxOffset {
			cursorOffset = maxOffset
		}
		if cursorOffset < int(file.Pos(0)) {
			cursorOffset = int(file.Pos(0))
		}
	}
	pos := file.Pos(cursorOffset)
	if !pos.IsValid() {
		return token.NoPos, fmt.Errorf("failed to calculate valid token.Pos for offset %d", cursorOffset)
	}
	return pos, nil
}

// addSignatureToScope adds parameters and named results from a function signature to the scope map (internal).
func addSignatureToScope(sig *types.Signature, scope map[string]types.Object) {
	if sig == nil {
		return
	}
	params := sig.Params()
	if params != nil {
		for j := 0; j < params.Len(); j++ {
			param := params.At(j)
			if param != nil && param.Name() != "" {
				if _, exists := scope[param.Name()]; !exists {
					scope[param.Name()] = param
				}
			}
		}
	}
	results := sig.Results()
	if results != nil {
		for j := 0; j < results.Len(); j++ {
			res := results.At(j)
			if res != nil && res.Name() != "" {
				if _, exists := scope[res.Name()]; !exists {
					scope[res.Name()] = res
				}
			}
		}
	}
}

// addScopeVariables adds variables from a types.Scope to the scope map (internal).
func addScopeVariables(typeScope *types.Scope, cursorPos token.Pos, scopeMap map[string]types.Object) {
	if typeScope == nil {
		return
	}
	for _, name := range typeScope.Names() {
		obj := typeScope.Lookup(name)
		include := !cursorPos.IsValid() || !obj.Pos().IsValid() || obj.Pos() < cursorPos
		if obj != nil && include {
			switch obj.(type) {
			case *types.Var, *types.Const, *types.TypeName, *types.Func, *types.Label, *types.PkgName, *types.Builtin:
				if _, exists := scopeMap[name]; !exists {
					scopeMap[name] = obj
				}
			case *types.Nil:
				if _, exists := scopeMap[name]; !exists {
					scopeMap[name] = obj
				}
			}
		}
	}
}

// formatFuncSignature creates a string representation from an ast.FuncDecl (fallback, internal).
func formatFuncSignature(f *ast.FuncDecl) string {
	var sb strings.Builder
	sb.WriteString("func ")
	if f.Recv != nil && len(f.Recv.List) > 0 {
		sb.WriteString("(...) ")
	}
	if f.Name != nil {
		sb.WriteString(f.Name.Name)
	} else {
		sb.WriteString("[anonymous]")
	}
	sb.WriteString("(...)")
	if f.Type != nil && f.Type.Results != nil {
		sb.WriteString(" (...)")
	}
	return sb.String()
}

// findCommentsWithMap uses ast.CommentMap to find comments near the cursor or associated with enclosing nodes (internal).
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		return comments
	}
	cursorLine := fset.Position(cursorPos).Line
	foundPreceding := false
	var precedingComments []string
	for node := range cmap {
		if node == nil {
			continue
		}
		for _, cg := range cmap[node] {
			if cg == nil {
				continue
			}
			commentEndLine := fset.Position(cg.End()).Line
			if commentEndLine == cursorLine-1 {
				for _, c := range cg.List {
					if c != nil {
						precedingComments = append(precedingComments, c.Text)
					}
				}
				foundPreceding = true
				break
			}
		}
		if foundPreceding {
			break
		}
	}
	if foundPreceding {
		comments = append(comments, precedingComments...)
		seen := make(map[string]struct{})
		uniqueComments := make([]string, 0, len(comments))
		for _, c := range comments {
			if _, ok := seen[c]; !ok {
				seen[c] = struct{}{}
				uniqueComments = append(uniqueComments, c)
			}
		}
		return uniqueComments
	}
	if path != nil {
		for i := 0; i < len(path); i++ {
			node := path[i]
			var docComment *ast.CommentGroup
			switch n := node.(type) {
			case *ast.FuncDecl:
				docComment = n.Doc
			case *ast.GenDecl:
				docComment = n.Doc
			case *ast.TypeSpec:
				docComment = n.Doc
			case *ast.Field:
				docComment = n.Doc
			case *ast.ValueSpec:
				docComment = n.Doc
			}
			if docComment != nil {
				for _, c := range docComment.List {
					if c != nil {
						comments = append(comments, c.Text)
					}
				}
				goto cleanup
			}
		}
	}
	if path != nil {
		for i := 0; i < len(path); i++ {
			node := path[i]
			if cgs := cmap.Filter(node).Comments(); len(cgs) > 0 {
				for _, cg := range cgs {
					for _, c := range cg.List {
						if c != nil {
							comments = append(comments, c.Text)
						}
					}
				}
				break
			}
		}
	}
cleanup:
	seen := make(map[string]struct{})
	uniqueComments := make([]string, 0, len(comments))
	for _, c := range comments {
		if _, ok := seen[c]; !ok {
			seen[c] = struct{}{}
			uniqueComments = append(uniqueComments, c)
		}
	}
	return uniqueComments
}

// findContextNodes tries to find the identifier, selector expression, or call expression
// at/before the cursor and resolve its type information, updating the AstContextInfo struct directly.
func findContextNodes(path []ast.Node, cursorPos token.Pos, pkg *packages.Package, info *AstContextInfo) {
	if len(path) == 0 {
		log.Println("Cannot find context nodes: Missing AST path.")
		return
	}
	hasTypeInfo := pkg != nil && pkg.TypesInfo != nil
	innermostNode := path[0]

	// Check CallExpr
	if callExpr, ok := innermostNode.(*ast.CallExpr); ok && cursorPos > callExpr.Lparen && cursorPos <= callExpr.Rparen {
		info.CallExpr = callExpr
		info.CallArgIndex = 0
		for i, arg := range callExpr.Args {
			if cursorPos > arg.End() {
				info.CallArgIndex = i + 1
			} else {
				break
			}
		}
		log.Printf("Cursor likely at argument index %d", info.CallArgIndex)
		if hasTypeInfo {
			if pkg.TypesInfo.Types == nil {
				addAnalysisError(info, errors.New("types map missing for call expr"))
				return
			}
			if tv, ok := pkg.TypesInfo.Types[callExpr.Fun]; ok && tv.IsValue() {
				if sig, ok := tv.Type.Underlying().(*types.Signature); ok {
					info.CallExprFuncType = sig
				} else {
					addAnalysisError(info, fmt.Errorf("type of call func is %T, not signature", tv.Type))
				}
			} else {
				addAnalysisError(info, fmt.Errorf("missing type info for call func %T", callExpr.Fun))
			}
		}
		return
	}

	// Check SelectorExpr
	for i := 0; i < len(path) && i < 2; i++ {
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok && (cursorPos > selExpr.X.End() || cursorPos == selExpr.X.End()) {
			info.SelectorExpr = selExpr
			if hasTypeInfo {
				if pkg.TypesInfo.Types == nil {
					addAnalysisError(info, errors.New("types map missing for selector expr"))
					return
				}
				if tv, ok := pkg.TypesInfo.Types[selExpr.X]; ok {
					info.SelectorExprType = tv.Type
					if tv.Type == nil {
						addAnalysisError(info, fmt.Errorf("missing type for selector base %T", selExpr.X))
					}
				} else {
					addAnalysisError(info, fmt.Errorf("missing type info for selector base %T", selExpr.X))
				}
			}
			return
		}
	}

	// Check Identifier
	var ident *ast.Ident
	for _, node := range path {
		if id, ok := node.(*ast.Ident); ok && id.End() == cursorPos {
			ident = id
			break
		}
	}
	if ident == nil {
		log.Println("No specific identifier found ending at cursor position.")
		return
	}
	info.IdentifierAtCursor = ident
	if !hasTypeInfo {
		log.Println("Skipping identifier type resolution due to missing type info.")
		return
	}
	if pkg.TypesInfo.Uses == nil && pkg.TypesInfo.Defs == nil && pkg.TypesInfo.Types == nil {
		addAnalysisError(info, errors.New("type info maps nil"))
		return
	}
	var obj types.Object
	var typ types.Type
	if pkg.TypesInfo.Uses != nil {
		if o := pkg.TypesInfo.Uses[ident]; o != nil {
			obj = o
			typ = o.Type()
		}
	}
	if obj == nil && pkg.TypesInfo.Defs != nil {
		if o := pkg.TypesInfo.Defs[ident]; o != nil {
			obj = o
			typ = o.Type()
		}
	}
	if obj == nil && pkg.TypesInfo.Implicits != nil {
		if o := pkg.TypesInfo.Implicits[ident]; o != nil {
			obj = o
			typ = o.Type()
		}
	}
	if obj == nil && pkg.TypesInfo.Types != nil {
		if tv, ok := pkg.TypesInfo.Types[ident]; ok && tv.Type != nil {
			typ = tv.Type
		} else {
			addAnalysisError(info, fmt.Errorf("missing type info for identifier '%s'", ident.Name))
		}
	}
	if obj != nil && typ == nil {
		addAnalysisError(info, fmt.Errorf("object '%s' found but type is nil", obj.Name()))
	}
	info.IdentifierObject = obj
	info.IdentifierType = typ
}

// listTypeMembers attempts to list fields and methods for a given type.
func listTypeMembers(typ types.Type, pkg *packages.Package) []string {
	if typ == nil {
		return nil
	}
	var members []string
	var qualifier types.Qualifier
	// Defend against nil pkg or pkg.Types
	if pkg != nil && pkg.Types != nil {
		qualifier = types.RelativeTo(pkg.Types)
	}

	currentType := typ
	isPointer := false
	if ptr, ok := typ.(*types.Pointer); ok {
		currentType = ptr.Elem()
		isPointer = true
	}
	if currentType == nil {
		return nil
	} // Handle nil element type

	underlying := currentType.Underlying()
	if underlying == nil {
		return nil
	} // Handle nil underlying type

	// Handle Basic types (just return the type name)
	if basic, ok := underlying.(*types.Basic); ok {
		return []string{basic.String()}
	}
	if _, ok := underlying.(*types.Map); ok {
		return []string{"// map type"}
	}
	if _, ok := underlying.(*types.Slice); ok {
		return []string{"// slice type"}
	}
	if _, ok := underlying.(*types.Chan); ok {
		return []string{"// channel type"}
	}

	if st, ok := underlying.(*types.Struct); ok {
		for i := 0; i < st.NumFields(); i++ {
			field := st.Field(i)
			if field != nil && field.Exported() {
				members = append(members, types.ObjectString(field, qualifier))
			}
		}
	}
	if iface, ok := underlying.(*types.Interface); ok {
		for i := 0; i < iface.NumExplicitMethods(); i++ {
			method := iface.ExplicitMethod(i)
			if method != nil && method.Exported() {
				members = append(members, types.ObjectString(method, qualifier))
			}
		}
		for i := 0; i < iface.NumEmbeddeds(); i++ {
			embeddedType := iface.EmbeddedType(i)
			if embeddedType != nil {
				members = append(members, fmt.Sprintf("// embeds %s", types.TypeString(embeddedType, qualifier)))
			}
		}
	}

	// Get method set for the original type (could be pointer or non-pointer)
	mset := types.NewMethodSet(typ)
	for i := 0; i < mset.Len(); i++ {
		sel := mset.At(i)
		if sel != nil {
			methodObj := sel.Obj()
			// Defensive check for method object type
			if method, ok := methodObj.(*types.Func); ok {
				if method != nil && method.Exported() {
					members = append(members, types.ObjectString(method, qualifier))
				}
			} else if methodObj != nil { // Log if it's not nil but also not a *types.Func
				log.Printf("Warning: Object in method set is not *types.Func: %T", methodObj)
			}
		}
	}

	// If original type was not a pointer, but underlying was named, also check pointer method set
	if !isPointer {
		if named, ok := currentType.(*types.Named); ok {
			// Check if named is not nil before creating pointer
			if named != nil {
				ptrToNamed := types.NewPointer(named)
				msetPtr := types.NewMethodSet(ptrToNamed)
				for i := 0; i < msetPtr.Len(); i++ {
					sel := msetPtr.At(i)
					if sel != nil {
						methodObj := sel.Obj()
						if method, ok := methodObj.(*types.Func); ok {
							if method != nil && method.Exported() {
								members = append(members, types.ObjectString(method, qualifier))
							}
						} else if methodObj != nil {
							log.Printf("Warning: Object in base method set is not *types.Func: %T", methodObj)
						}
					}
				}
			}
		}
	}

	if len(members) > 0 {
		seen := make(map[string]struct{})
		uniqueMembers := make([]string, 0, len(members))
		for _, m := range members {
			if _, ok := seen[m]; !ok {
				seen[m] = struct{}{}
				uniqueMembers = append(uniqueMembers, m)
			}
		}
		members = uniqueMembers
	}
	return members
}

// logAnalysisErrors logs any non-fatal errors found during context analysis (internal).
func logAnalysisErrors(errs []error) {
	if len(errs) > 0 {
		log.Printf("Context analysis completed with %d non-fatal error(s):", len(errs))
		// Use errors.Join to format potentially nested errors nicely
		log.Printf("  Analysis Errors: %v", errors.Join(errs...))
	}
}

// addAnalysisError is a helper to log and store non-fatal analysis errors.
func addAnalysisError(info *AstContextInfo, err error) {
	if err != nil && info != nil { // Check info is not nil
		log.Printf("Analysis Warning: %v", err)
		info.AnalysisErrors = append(info.AnalysisErrors, err)
	}
}

// =============================================================================
// Spinner & File Helpers (Spinner Exported, others internal)
// =============================================================================

// Spinner for terminal feedback. Exported Type.
type Spinner struct {
	chars    []string
	message  string // Current message to display
	index    int
	mu       sync.Mutex // Protects message and index
	stopChan chan struct{}
	doneChan chan struct{} // Used for graceful shutdown confirmation
	running  bool
}

// NewSpinner creates a new Spinner. Exported Function.
func NewSpinner() *Spinner {
	return &Spinner{
		chars: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}, // Extended chars
		index: 0,
		// stopChan and doneChan initialized in Start
	}
}

// Start starts the spinner animation in a goroutine with an initial message. Exported Method.
func (s *Spinner) Start(initialMessage string) {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return // Already running
	}
	// Ensure channels are fresh if Start is called after Stop
	s.stopChan = make(chan struct{})
	s.doneChan = make(chan struct{}) // Initialize doneChan
	s.message = initialMessage       // Set initial message
	s.running = true
	s.mu.Unlock()

	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		// Signal completion and cleanup when goroutine exits
		defer func() {
			fmt.Print("\r\033[K") // Clear the line completely on stop
			s.mu.Lock()
			s.running = false
			s.mu.Unlock()
			close(s.doneChan) // Signal that the goroutine has finished cleanup
		}()

		for {
			select {
			case <-s.stopChan:
				return // Exit goroutine
			case <-ticker.C:
				s.mu.Lock()
				// Check if running flag is still true before proceeding
				if !s.running {
					s.mu.Unlock()
					return
				}
				char := s.chars[s.index]
				msg := s.message // Read current message under lock
				s.index = (s.index + 1) % len(s.chars)
				// Use carriage return (\r) and clear line (\033[K)
				fmt.Printf("\r\033[K%s%s%s %s", ColorCyan, char, ColorReset, msg) // Use exported constant
				s.mu.Unlock()
			}
		}
	}()
}

// UpdateMessage updates the message displayed next to the spinner. Exported Method.
func (s *Spinner) UpdateMessage(newMessage string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Only update if spinner is actually running
	if s.running {
		s.message = newMessage
	}
}

// Stop stops the spinner animation and waits for cleanup. Exported Method.
func (s *Spinner) Stop() {
	s.mu.Lock()
	// Check running before trying to close channel
	if !s.running {
		s.mu.Unlock()
		return
	}
	// Close stopChan safely if not already closed
	select {
	case <-s.stopChan: // Already closed
	default:
		close(s.stopChan)
	}
	doneChan := s.doneChan // Read doneChan under lock
	s.mu.Unlock()          // Unlock before waiting

	// Wait for the goroutine to signal completion (with a timeout)
	if doneChan != nil {
		select {
		case <-doneChan:
			// Goroutine finished cleanup
		case <-time.After(250 * time.Millisecond): // Increased timeout slightly
			log.Println("Warning: Timeout waiting for spinner goroutine cleanup")
		}
	}
}

// extractSnippetContext reads file content and returns prefix, suffix, and full line based on cursor (internal).
func extractSnippetContext(filename string, row, col int) (SnippetContext, error) {
	var ctx SnippetContext
	contentBytes, err := os.ReadFile(filename)
	if err != nil {
		return ctx, fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	content := string(contentBytes)
	fset := token.NewFileSet()
	absFilename, _ := filepath.Abs(filename)
	if absFilename == "" {
		absFilename = "input.go"
	}
	file := fset.AddFile(absFilename, fset.Base(), len(contentBytes))
	if file == nil {
		return ctx, fmt.Errorf("failed to add file '%s' to fileset", absFilename)
	}
	cursorPos, posErr := calculateCursorPos(file, row, col)
	if posErr != nil {
		return ctx, fmt.Errorf("cannot determine valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		return ctx, fmt.Errorf("invalid cursor position calculated")
	}
	offset := file.Offset(cursorPos)
	if offset < 0 || offset > len(content) {
		return ctx, fmt.Errorf("calculated offset %d is out of bounds [0, %d]", offset, len(content))
	}
	ctx.Prefix = content[:offset]
	ctx.Suffix = content[offset:]
	lineStartPos := file.LineStart(row)
	lineEndOffset := -1
	if row < file.LineCount() {
		lineEndPos := file.LineStart(row + 1)
		if lineEndPos.IsValid() {
			lineEndOffset = file.Offset(lineEndPos)
		}
	}
	if lineEndOffset == -1 {
		lineEndOffset = file.Size()
	}
	startOffset := file.Offset(lineStartPos)
	if startOffset >= 0 && lineEndOffset >= startOffset && lineEndOffset <= len(content) {
		ctx.FullLine = content[startOffset:lineEndOffset]
		ctx.FullLine = strings.TrimSuffix(ctx.FullLine, "\n")
		ctx.FullLine = strings.TrimSuffix(ctx.FullLine, "\r")
	} else {
		log.Printf("Warning: Could not extract full line for row %d", row)
	}
	return ctx, nil
}

// readComments is kept but less critical now as comments are handled via AST analysis (internal).
func readComments(code string) []string {
	var comments []string
	lines := strings.Split(code, "\n")
	inBlockComment := false
	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "/*") {
			inBlockComment = true
			commentContent := strings.TrimPrefix(trimmedLine, "/*")
			if strings.HasSuffix(commentContent, "*/") {
				commentContent = strings.TrimSuffix(commentContent, "*/")
				inBlockComment = false
			}
			if comment := strings.TrimSpace(commentContent); comment != "" {
				comments = append(comments, comment)
			}
			continue
		}
		if inBlockComment {
			commentContent := trimmedLine
			if strings.HasSuffix(trimmedLine, "*/") {
				commentContent = strings.TrimSuffix(trimmedLine, "*/")
				inBlockComment = false
			}
			if comment := strings.TrimSpace(commentContent); comment != "" {
				comments = append(comments, comment)
			}
			continue
		}
		if strings.HasPrefix(trimmedLine, "//") {
			comment := strings.TrimSpace(strings.TrimPrefix(trimmedLine, "//"))
			if comment != "" {
				comments = append(comments, comment)
			}
		}
	}
	return comments
}
