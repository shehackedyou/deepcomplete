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
	"go/format" // For formatting receiver in preamble
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
Analyze the provided context (enclosing function/method, imports, scope variables, comments, code structure) and the preceding code snippet.
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

// Validate checks if the configuration values are valid.
func (c *Config) Validate() error {
	if strings.TrimSpace(c.OllamaURL) == "" {
		return errors.New("ollama_url cannot be empty")
	}
	if _, err := url.ParseRequestURI(c.OllamaURL); err != nil {
		return fmt.Errorf("invalid ollama_url: %w", err)
	}
	if strings.TrimSpace(c.Model) == "" {
		return errors.New("model cannot be empty")
	}
	if c.MaxTokens <= 0 {
		log.Printf("Warning: max_tokens (%d) is not positive, using default %d",
			c.MaxTokens, defaultMaxTokens)
		c.MaxTokens = defaultMaxTokens
	}
	if c.Temperature < 0 {
		log.Printf("Warning: temperature (%.2f) is negative, using default %.2f",
			c.Temperature, defaultTemperature)
		c.Temperature = defaultTemperature
	}
	return nil
}

// FileConfig represents the structure of the JSON config file.
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
	EnclosingFunc      *types.Func   // Function or method type object
	EnclosingFuncNode  *ast.FuncDecl // Function or method AST node
	ReceiverType       string        // Formatted receiver type if it's a method
	EnclosingBlock     *ast.BlockStmt
	Imports            []*ast.ImportSpec
	CommentsNearCursor []string
	IdentifierAtCursor *ast.Ident
	IdentifierType     types.Type
	IdentifierObject   types.Object
	SelectorExpr       *ast.SelectorExpr
	SelectorExprType   types.Type
	CallExpr           *ast.CallExpr
	CallExprFuncType   *types.Signature
	CallArgIndex       int
	ExpectedArgType    types.Type
	CompositeLit       *ast.CompositeLit
	CompositeLitType   types.Type
	VariablesInScope   map[string]types.Object
	PromptPreamble     string
	AnalysisErrors     []error
}

// SnippetContext holds the code prefix and suffix for prompting.
type SnippetContext struct {
	Prefix   string
	Suffix   string
	FullLine string // The full line where the cursor is located
}

// OllamaError defines a custom error for Ollama API issues.
type OllamaError struct {
	Message string
	Status  int
}

// Error implements the error interface for OllamaError.
func (e *OllamaError) Error() string {
	return fmt.Sprintf("Ollama error: %s (Status: %d)", e.Message, e.Status)
}

// OllamaResponse represents the streaming response structure.
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"`
}

// =============================================================================
// Exported Errors
// =============================================================================

var (
	ErrAnalysisFailed    = errors.New("code analysis failed")
	ErrOllamaUnavailable = errors.New("ollama API unavailable or returned server error")
	ErrStreamProcessing  = errors.New("error processing LLM stream")
	ErrConfig            = errors.New("configuration error")
	ErrInvalidConfig     = errors.New("invalid configuration")
)

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
		PromptTemplate: promptTemplate,
		FimTemplate:    fimPromptTemplate,
		MaxTokens:      defaultMaxTokens,
		Stop:           []string{DefaultStop, "}", "//", "/*"},
		Temperature:    defaultTemperature,
		UseAst:         true,
		UseFim:         false,
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
func LoadConfig() (Config, error) {
	cfg := DefaultConfig // Start with defaults
	var loadedFromFile bool
	var loadErrors []error

	primaryPath, secondaryPath, pathErr := getConfigPaths()
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		log.Printf("Warning: Could not determine config paths: %v", pathErr)
	}

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

	if !loadedFromFile {
		log.Println("No valid config file found in standard locations.")
		writePath := primaryPath
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

	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	// Return combined non-fatal errors wrapped in ErrConfig
	if len(loadErrors) > 0 {
		return cfg, fmt.Errorf("%w: %w", ErrConfig, errors.Join(loadErrors...))
	}

	return cfg, nil
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
		return true, fmt.Errorf("reading config file %q failed: %w", path, err)
	}
	if len(data) == 0 {
		log.Printf("Warning: Config file exists but is empty: %s", path)
		return true, nil // Treat as loaded but with no overrides
	}
	var fileCfg FileConfig
	if err := json.Unmarshal(data, &fileCfg); err != nil {
		return true, fmt.Errorf("parsing config file JSON %q failed: %w", path, err)
	}
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
	if err := os.MkdirAll(dir, 0750); err != nil {
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}
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
	if err := os.WriteFile(path, jsonData, 0640); err != nil {
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
			return nil, fmt.Errorf("%w: ollama request timed out: %w", ErrOllamaUnavailable, err)
		}
		return nil, fmt.Errorf("%w: error making HTTP request: %w", ErrOllamaUnavailable, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(read error)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error
			}
		}
		err = &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
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
	info = &AstContextInfo{
		FilePath:         filename,
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		CallArgIndex:     -1,
	}
	defer func() {
		r := recover()
		if r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			addAnalysisError(info, panicErr)
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				analysisErr = errors.Join(analysisErr, panicErr)
			}
			log.Printf("Panic recovered during AnalyzeCodeContext: %v\n%s", r, string(debug.Stack()))
		}
	}()

	absFilename, err := filepath.Abs(filename)
	if err != nil {
		return info, fmt.Errorf("failed to get absolute path for '%s': %w", filename, err)
	}
	info.FilePath = absFilename
	log.Printf("Starting context analysis for: %s (%d:%d)", absFilename, line, col)

	fset := token.NewFileSet()
	targetPkg, targetFileAST, targetFile, loadErrors := loadPackageInfo(ctx, absFilename, fset)
	for _, loadErr := range loadErrors {
		addAnalysisError(info, loadErr)
	}

	if targetFileAST == nil {
		log.Printf("Attempting direct parse of %s", absFilename)
		targetFileAST, targetFile, err = directParse(absFilename, fset)
		if err != nil {
			addAnalysisError(info, err)
			analysisErr = errors.Join(info.AnalysisErrors...)
			return info, fmt.Errorf("%w: direct parse failed: %w", ErrAnalysisFailed, analysisErr)
		}
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

	if targetFileAST != nil {
		info.Imports = targetFileAST.Imports
	}

	// Perform analysis steps only if we have a valid file token
	if targetFile != nil {
		err = a.performAnalysisSteps(targetFile, targetFileAST, targetPkg, fset, line, col, info)
		if err != nil {
			addAnalysisError(info, err)
		} // Add non-fatal error from analysis steps
	} else {
		addAnalysisError(info, errors.New("cannot perform analysis steps: missing token.File"))
	}

	info.PromptPreamble = buildPreamble(info, targetPkg)
	log.Printf("Generated Context Preamble:\n---\n%s\n---", info.PromptPreamble)
	logAnalysisErrors(info.AnalysisErrors) // Log all collected non-fatal errors

	analysisErr = errors.Join(info.AnalysisErrors...)
	// Wrap the combined analysis errors only if they exist
	if analysisErr != nil {
		return info, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisErr)
	}
	return info, nil
}

// performAnalysisSteps encapsulates the core analysis logic after loading/parsing.
func (a *GoPackagesAnalyzer) performAnalysisSteps(
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package,
	fset *token.FileSet,
	line, col int,
	info *AstContextInfo,
) error {
	// This function now assumes targetFile is not nil
	cursorPos, posErr := calculateCursorPos(targetFile, line, col)
	if posErr != nil {
		return fmt.Errorf("cursor position error: %w", posErr) // Return fatal error
	}
	info.CursorPos = cursorPos
	log.Printf("Calculated cursor token.Pos: %d (Line: %d, Col: %d)",
		info.CursorPos, fset.Position(info.CursorPos).Line, fset.Position(info.CursorPos).Column)

	path := findEnclosingPath(targetFileAST, info.CursorPos, info)
	if path != nil {
		findContextNodes(path, info.CursorPos, targetPkg, info)
	}
	gatherScopeContext(path, targetPkg, info)
	findRelevantComments(targetFileAST, path, info.CursorPos, fset, info)
	return nil // Non-fatal errors are added to info.AnalysisErrors
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
	snippet := snippetCtx.Prefix

	if config.UseFim {
		template = config.FimTemplate
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix
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
		finalPrompt = fmt.Sprintf(template, prefix, suffix)
		finalPrompt = contextPreamble + "\n" + finalPrompt
	} else {
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

// NewDeepCompleter creates a new DeepCompleter service with default components. Exported.
func NewDeepCompleter() (*DeepCompleter, error) {
	cfg, configErr := LoadConfig()
	if configErr != nil && !errors.Is(configErr, ErrConfig) { // Handle fatal config path errors
		return nil, configErr
	} else if configErr != nil { // Log non-fatal config load/write warnings
		log.Printf("Warning during config load: %v", configErr)
		cfg = DefaultConfig // Ensure defaults on non-fatal load error
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	return &DeepCompleter{client: newHttpOllamaClient(), analyzer: NewGoPackagesAnalyzer(), formatter: newTemplateFormatter(), config: cfg}, nil
}

// NewDeepCompleterWithConfig creates a new DeepCompleter service with provided config. Exported.
func NewDeepCompleterWithConfig(config Config) (*DeepCompleter, error) {
	if config.PromptTemplate == "" {
		config.PromptTemplate = promptTemplate
	}
	if config.FimTemplate == "" {
		config.FimTemplate = fimPromptTemplate
	}
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	return &DeepCompleter{client: newHttpOllamaClient(), analyzer: NewGoPackagesAnalyzer(), formatter: newTemplateFormatter(), config: config}, nil
}

// GetCompletion provides completion for a raw code snippet. Non-streaming. Exported.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	log.Println("DeepCompleter.GetCompletion called for basic prompt.")
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, dc.config)
	reader, err := dc.client.GenerateStream(ctx, prompt, dc.config)
	if err != nil {
		var ollamaErr *OllamaError
		if errors.As(err, &ollamaErr) || errors.Is(err, ErrOllamaUnavailable) {
			return "", fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		return "", fmt.Errorf("failed to call Ollama API for basic prompt: %w", err)
	}
	var buffer bytes.Buffer
	streamCtx, cancelStream := context.WithTimeout(ctx, 50*time.Second)
	defer cancelStream()
	if streamErr := streamCompletion(streamCtx, reader, &buffer); streamErr != nil {
		return "", fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
	}
	return buffer.String(), nil
}

// GetCompletionStreamFromFile uses AST context and streams completion. Exported.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, filename string, row, col int, w io.Writer) error {
	var contextPreamble string = "// Basic file context only."
	var analysisErr error // Stores combined non-fatal errors from analysis

	if dc.config.UseAst {
		log.Printf("Analyzing context using AST/Types for %s:%d:%d", filename, row, col)
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 20*time.Second)
		var analysisInfo *AstContextInfo
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, filename, row, col)
		cancelAnalysis()

		// If Analyze itself failed fatally, return the error immediately
		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		// Log non-fatal errors collected during analysis if Analyze didn't fail fatally
		if analysisErr != nil { // This means errors.Is(analysisErr, ErrAnalysisFailed) is true
			logAnalysisErrors(analysisInfo.AnalysisErrors) // Log the collected errors
		}

		if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else {
			contextPreamble += "// Warning: AST analysis returned no specific context.\n"
		}
	} else {
		log.Println("AST analysis disabled.")
	}

	snippetCtx, err := extractSnippetContext(filename, row, col)
	if err != nil {
		return fmt.Errorf("failed to extract code snippet context: %w", err)
	}

	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, dc.config)
	log.Printf("Generated Prompt (length %d):\n---\n%s\n---", len(prompt), prompt)

	apiCallFunc := func() error {
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second)
		defer cancelApi()
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, dc.config)
		if apiErr != nil {
			var oe *OllamaError
			if errors.As(apiErr, &oe) {
				return apiErr
			} // Let retry handle OllamaError
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, apiErr) // Wrap others
		}
		PrettyPrint(ColorGreen, "Completion:\n")
		streamErr := streamCompletion(apiCtx, reader, w)
		fmt.Fprintln(w) // Final newline
		if streamErr != nil {
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr) // Wrap stream errors
		}
		log.Println("Completion stream finished successfully for this attempt.")
		return nil
	}

	err = retry(ctx, apiCallFunc, maxRetries, retryDelay)
	if err != nil {
		// Wrap retry final error appropriately
		var oe *OllamaError
		if errors.As(err, &oe) || errors.Is(err, ErrOllamaUnavailable) {
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
		}
		return fmt.Errorf("failed to get completion stream after retries: %w", err)
	}

	// Log analysis errors again as a final warning, even if completion succeeded
	// Note: analysisErr here contains non-fatal errors wrapped with ErrAnalysisFailed if any occurred.
	if analysisErr != nil {
		log.Printf("Warning: Completion succeeded, but context analysis encountered non-fatal errors: %v", analysisErr)
	}
	return nil
}

// =============================================================================
// Internal Helpers (Unexported package-level functions)
// =============================================================================

// --- Ollama Stream Processing Helpers ---
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
				return nil // Successful end of stream
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
func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil
	}
	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		log.Printf("Error unmarshalling Ollama JSON line: %v, line content: '%s'", err, string(line))
		return nil // Continue processing next line
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
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration) error {
	var err error
	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		err = operation()
		if err == nil {
			return nil // Success
		}
		var ollamaErr *OllamaError
		isRetryable := errors.As(err, &ollamaErr) &&
			(ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		isRetryable = isRetryable || errors.Is(err, ErrOllamaUnavailable) // Also retry on wrapped ErrOllamaUnavailable

		if !isRetryable {
			log.Printf("Non-retryable error encountered: %v", err)
			return err // Return non-retryable errors immediately
		}
		log.Printf("Attempt %d failed with retryable error: %v. Retrying in %v...", i+1, err, currentDelay)
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled during retry wait: %v", ctx.Err())
			return ctx.Err() // Respect context cancellation
		case <-time.After(currentDelay):
			// Optional: Implement exponential backoff
			// currentDelay *= 2
		}
	}
	log.Printf("Operation failed after %d retries.", maxRetries)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, err) // Return the last error
}

// --- Analysis Helpers ---
func loadPackageInfo(ctx context.Context, absFilename string, fset *token.FileSet) (*packages.Package, *ast.File, *token.File, []error) {
	dir := filepath.Dir(absFilename)
	var loadErrors []error
	cfg := &packages.Config{
		Context: ctx,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedSyntax |
			packages.NeedTypesInfo | packages.NeedTypesSizes | packages.NeedDeps,
		Dir:  dir,
		Fset: fset,
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			const mode = parser.ParseComments | parser.AllErrors
			file, err := parser.ParseFile(fset, filename, src, mode)
			if err != nil {
				log.Printf("Parser error in %s (ignored for partial AST): %v", filename, err)
			}
			return file, nil
		},
		Tests: true,
	}
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
		// **FIX:** Correctly use blank identifier for unused filePath variable
		for i, _ := range pkg.CompiledGoFiles { // Use _ for filePath
			if i >= len(pkg.Syntax) {
				continue // Avoid index out of range
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
func directParse(absFilename string, fset *token.FileSet) (*ast.File, *token.File, error) {
	srcBytes, readErr := os.ReadFile(absFilename)
	if readErr != nil {
		return nil, nil, fmt.Errorf("direct read failed for '%s': %w", absFilename, readErr)
	}
	const mode = parser.ParseComments | parser.AllErrors
	targetFileAST, parseErr := parser.ParseFile(fset, absFilename, srcBytes, mode)
	if parseErr != nil {
		log.Printf("Direct parsing of %s failed: %v", absFilename, parseErr)
		// Continue even if there are parse errors, as long as AST is not nil
	}
	if targetFileAST == nil {
		return nil, nil, errors.New("failed to obtain any AST from direct parse")
	}
	targetFile := fset.File(targetFileAST.Pos())
	return targetFileAST, targetFile, parseErr // Return parseErr as non-fatal
}
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
func gatherScopeContext(path []ast.Node, targetPkg *packages.Package, info *AstContextInfo) {
	if path == nil {
		return
	}
	var fset *token.FileSet
	if targetPkg != nil && targetPkg.Fset != nil {
		fset = targetPkg.Fset
	} else {
		fset = token.NewFileSet()
		log.Println("Warning: Using fallback FileSet for receiver formatting.")
	}

	for i := len(path) - 1; i >= 0; i-- {
		node := path[i]
		switch n := node.(type) {
		case *ast.FuncDecl:
			if info.EnclosingFuncNode == nil {
				info.EnclosingFuncNode = n
				if n.Recv != nil && len(n.Recv.List) > 0 && n.Recv.List[0].Type != nil {
					var buf bytes.Buffer
					if err := format.Node(&buf, fset, n.Recv.List[0].Type); err == nil {
						info.ReceiverType = buf.String()
					} else {
						log.Printf("Warning: could not format receiver type: %v", err)
						info.ReceiverType = "[error formatting receiver]"
					}
				}
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
func buildPreamble(info *AstContextInfo, targetPkg *packages.Package) string {
	var preamble strings.Builder
	qualifier := types.RelativeTo(targetPkg.Types) // Handle nil pkg
	preamble.WriteString(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), info.PackageName))
	formatImportsSection(&preamble, info)
	formatEnclosingFuncSection(&preamble, info, qualifier)
	formatCommentsSection(&preamble, info)
	formatCursorContextSection(&preamble, info, targetPkg, qualifier)
	formatScopeSection(&preamble, info, qualifier)
	return preamble.String()
}
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo) {
	if len(info.Imports) > 0 {
		preamble.WriteString("// Imports:\n")
		for _, imp := range info.Imports {
			if imp == nil {
				continue
			}
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
}
func formatEnclosingFuncSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier) {
	funcOrMethod := "Function"
	receiverStr := ""
	if info.ReceiverType != "" {
		funcOrMethod = "Method"
		receiverStr = fmt.Sprintf("(%s) ", info.ReceiverType)
	}
	if info.EnclosingFunc != nil {
		name := info.EnclosingFunc.Name()
		sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		if info.ReceiverType != "" && strings.HasPrefix(sigStr, "func(") {
			sigStr = "func" + strings.TrimPrefix(sigStr, "func")
		}
		preamble.WriteString(fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr))
	} else if info.EnclosingFuncNode != nil {
		name := "[anonymous]"
		if info.EnclosingFuncNode.Name != nil {
			name = info.EnclosingFuncNode.Name.Name
		}
		preamble.WriteString(fmt.Sprintf("// Enclosing %s (AST): %s%s(...)\n", funcOrMethod, receiverStr, name))
	}
}
func formatCommentsSection(preamble *strings.Builder, info *AstContextInfo) {
	if len(info.CommentsNearCursor) > 0 {
		preamble.WriteString("// Relevant Comments:\n")
		for _, c := range info.CommentsNearCursor {
			cleanComment := strings.TrimSpace(strings.TrimPrefix(c, "//"))
			cleanComment = strings.TrimSpace(strings.TrimPrefix(cleanComment, "/*"))
			cleanComment = strings.TrimSpace(strings.TrimSuffix(cleanComment, "*/"))
			preamble.WriteString(fmt.Sprintf("//   %s\n", cleanComment))
		}
	}
}
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, targetPkg *packages.Package, qualifier types.Qualifier) {
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
			if info.ExpectedArgType != nil {
				preamble.WriteString(fmt.Sprintf("// Expected Arg Type: %s\n", types.TypeString(info.ExpectedArgType, qualifier)))
			}
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
	} else if info.CompositeLit != nil && info.CompositeLitType != nil {
		typeName := types.TypeString(info.CompositeLitType, qualifier)
		preamble.WriteString(fmt.Sprintf("// Inside composite literal of type: %s\n", typeName))
		if st, ok := info.CompositeLitType.Underlying().(*types.Struct); ok {
			presentFields := make(map[string]bool)
			for _, elt := range info.CompositeLit.Elts {
				if kv, ok := elt.(*ast.KeyValueExpr); ok {
					if keyIdent, ok := kv.Key.(*ast.Ident); ok {
						presentFields[keyIdent.Name] = true
					}
				}
			}
			var missingFields []string
			for i := 0; i < st.NumFields(); i++ {
				field := st.Field(i)
				if field != nil && field.Exported() && !presentFields[field.Name()] {
					missingFields = append(missingFields, types.ObjectString(field, qualifier))
				}
			}
			if len(missingFields) > 0 {
				preamble.WriteString("//   Available fields to add:\n")
				sort.Strings(missingFields)
				for _, fieldStr := range missingFields {
					preamble.WriteString(fmt.Sprintf("//     - %s\n", fieldStr))
				}
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
}
func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier) {
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
}
func calculateCursorPos(file *token.File, line, col int) (token.Pos, error) {
	if line <= 0 {
		return token.NoPos, fmt.Errorf("invalid line number: %d (must be >= 1)", line)
	}
	if file == nil {
		return token.NoPos, errors.New("invalid token.File (nil)")
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
func findContextNodes(path []ast.Node, cursorPos token.Pos, pkg *packages.Package, info *AstContextInfo) {
	if len(path) == 0 {
		log.Println("Cannot find context nodes: Missing AST path.")
		return
	}
	hasTypeInfo := pkg != nil && pkg.TypesInfo != nil
	innermostNode := path[0]

	// Check CompositeLit first
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
		info.CompositeLit = compLit
		log.Printf("Cursor is inside CompositeLit at pos %d", compLit.Pos())
		if hasTypeInfo && pkg.TypesInfo.Types != nil {
			if tv, ok := pkg.TypesInfo.Types[compLit]; ok {
				info.CompositeLitType = tv.Type
				if info.CompositeLitType != nil {
					log.Printf("Resolved composite literal type: %s", types.TypeString(info.CompositeLitType, types.RelativeTo(pkg.Types)))
				} else {
					addAnalysisError(info, errors.New("composite literal type is nil"))
				}
			} else {
				addAnalysisError(info, errors.New("missing type info for composite literal"))
			}
		}
		return // Prioritize CompositeLit context
	}

	// Check CallExpr
	if callExpr, ok := innermostNode.(*ast.CallExpr); ok && cursorPos > callExpr.Lparen && cursorPos <= callExpr.Rparen {
		info.CallExpr = callExpr
		info.CallArgIndex = -1 // Initialize
		currentArgStart := callExpr.Lparen + 1
		for i, arg := range callExpr.Args {
			if cursorPos >= currentArgStart && cursorPos <= arg.End() {
				info.CallArgIndex = i
				break
			}
			currentArgStart = arg.End() + 1 // Position after this arg (assume comma exists)
			if i == len(callExpr.Args)-1 && cursorPos > arg.End() {
				info.CallArgIndex = i + 1
			}
		}
		if info.CallArgIndex == -1 && len(callExpr.Args) == 0 {
			info.CallArgIndex = 0
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
					params := sig.Params()
					if params != nil {
						idx := info.CallArgIndex
						var expectedType types.Type
						if sig.Variadic() && idx >= params.Len()-1 {
							lastParam := params.At(params.Len() - 1)
							if slice, ok := lastParam.Type().(*types.Slice); ok {
								expectedType = slice.Elem()
							}
						} else if idx >= 0 && idx < params.Len() { // Check idx validity
							expectedType = params.At(idx).Type()
						}
						info.ExpectedArgType = expectedType
					}
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
func listTypeMembers(typ types.Type, pkg *packages.Package) []string {
	if typ == nil {
		return nil
	}
	var members []string
	var qualifier types.Qualifier
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
	}
	underlying := currentType.Underlying()
	if underlying == nil {
		return nil
	}

	if basic, ok := underlying.(*types.Basic); ok {
		return []string{fmt.Sprintf("(type: %s)", basic.String())}
	}
	if _, ok := underlying.(*types.Map); ok {
		return []string{"// map operations: make, len, delete, range"}
	}
	if _, ok := underlying.(*types.Slice); ok {
		return []string{"// slice operations: make, len, cap, append, copy, range"}
	}
	if _, ok := underlying.(*types.Chan); ok {
		return []string{"// channel operations: make, len, cap, close, send (<-), receive (<-)"}
	}

	if st, ok := underlying.(*types.Struct); ok {
		for i := 0; i < st.NumFields(); i++ {
			field := st.Field(i)
			if field != nil && field.Exported() {
				members = append(members, fmt.Sprintf("field %s", types.ObjectString(field, qualifier)))
			}
		}
	}
	if iface, ok := underlying.(*types.Interface); ok {
		for i := 0; i < iface.NumExplicitMethods(); i++ {
			method := iface.ExplicitMethod(i)
			if method != nil && method.Exported() {
				members = append(members, fmt.Sprintf("method %s", types.ObjectString(method, qualifier)))
			}
		}
		for i := 0; i < iface.NumEmbeddeds(); i++ {
			embeddedType := iface.EmbeddedType(i)
			if embeddedType != nil {
				members = append(members, fmt.Sprintf("// embeds %s", types.TypeString(embeddedType, qualifier)))
			}
		}
	}
	mset := types.NewMethodSet(typ)
	for i := 0; i < mset.Len(); i++ {
		sel := mset.At(i)
		if sel != nil {
			methodObj := sel.Obj()
			if method, ok := methodObj.(*types.Func); ok {
				if method != nil && method.Exported() {
					members = append(members, fmt.Sprintf("method %s", types.ObjectString(method, qualifier)))
				}
			} else if methodObj != nil {
				log.Printf("Warning: Object in method set is not *types.Func: %T", methodObj)
			}
		}
	}
	if !isPointer {
		if named, ok := currentType.(*types.Named); ok {
			if named != nil {
				ptrToNamed := types.NewPointer(named)
				msetPtr := types.NewMethodSet(ptrToNamed)
				for i := 0; i < msetPtr.Len(); i++ {
					sel := msetPtr.At(i)
					if sel != nil {
						methodObj := sel.Obj()
						if method, ok := methodObj.(*types.Func); ok {
							if method != nil && method.Exported() {
								found := false
								for _, existing := range members {
									if strings.HasSuffix(existing, method.Name()+"(...)") || strings.Contains(existing, method.Name()+" func(") {
										found = true
										break
									}
								}
								if !found {
									members = append(members, fmt.Sprintf("method %s", types.ObjectString(method, qualifier)))
								}
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
func logAnalysisErrors(errs []error) {
	if len(errs) > 0 {
		log.Printf("Context analysis completed with %d non-fatal error(s):", len(errs))
		log.Printf("  Analysis Errors: %v", errors.Join(errs...))
	}
}
func addAnalysisError(info *AstContextInfo, err error) {
	if err != nil && info != nil {
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
		chars: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
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
	s.stopChan = make(chan struct{})
	s.doneChan = make(chan struct{})
	s.message = initialMessage
	s.running = true
	s.mu.Unlock()

	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		defer func() {
			fmt.Print("\r\033[K") // Clear line on stop
			s.mu.Lock()
			s.running = false
			s.mu.Unlock()
			select {
			case s.doneChan <- struct{}{}: // Signal done
			default: // Avoid blocking if Stop already timed out
			}
			close(s.doneChan)
		}()

		for {
			select {
			case <-s.stopChan:
				return
			case <-ticker.C:
				s.mu.Lock()
				if !s.running {
					s.mu.Unlock()
					return
				}
				char := s.chars[s.index]
				msg := s.message
				s.index = (s.index + 1) % len(s.chars)
				fmt.Printf("\r\033[K%s%s%s %s", ColorCyan, char, ColorReset, msg)
				s.mu.Unlock()
			}
		}
	}()
}

// UpdateMessage updates the message displayed next to the spinner. Exported Method.
func (s *Spinner) UpdateMessage(newMessage string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		s.message = newMessage
	}
}

// Stop stops the spinner animation and waits for cleanup. Exported Method.
func (s *Spinner) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	select {
	case <-s.stopChan: // Already closed
	default:
		close(s.stopChan)
	}
	doneChan := s.doneChan // Read under lock
	s.mu.Unlock()          // Unlock before waiting

	if doneChan != nil {
		select {
		case <-doneChan: // Wait for goroutine to signal done
		case <-time.After(300 * time.Millisecond):
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
	if lineEndOffset == -1 { // Last line
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
				inBlockComment = false // Single-line block comment
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
