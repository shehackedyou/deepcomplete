// deepcomplete/deepcomplete.go
package deepcomplete

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256" // For hashing
	"encoding/gob"  // For serializing cache data header struct ONLY
	"encoding/hex"
	"encoding/json"
	"errors" // Using errors.Join requires Go 1.20+
	"fmt"
	"go/ast"
	"go/format" // For formatting receiver in preamble
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"io/fs" // For file hashing errors
	"log"
	"net" // For network errors
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug" // For panic recovery stack trace
	"sort"
	"strings"
	"sync"
	"time"

	"go.etcd.io/bbolt"                      // Import bbolt
	"golang.org/x/tools/go/ast/astutil"    // Utility for AST traversal
	"golang.org/x/tools/go/packages"       // Standard way to load packages for analysis
	"golang.org/x/tools/go/types/exportdata" // For serializing types.Package (Step 25)
	// "golang.org/x/tools/go/types/typeutil" // May need for advanced type reconstruction
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
	defaultMaxTokens = 256
	// DefaultStop is exported for potential use in CLI default flags
	DefaultStop        = "\n" // Common stop sequences for code completion
	defaultTemperature = 0.1
	// Default config file name
	defaultConfigFileName = "config.json"
	configDirName         = "deepcomplete" // Subdirectory name for config/data

	// --- Cache Constants ---
	cacheSchemaVersion = 1 // To invalidate cache if format changes
)

var (
	// --- Cache Bucket Name ---
	cacheBucketName = []byte("PackageAnalysisCache")
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
	// --- Fields added in Step 8 ---
	MaxPreambleLen int      `json:"max_preamble_len"` // Max length for AST/Type context preamble
	MaxSnippetLen  int      `json:"max_snippet_len"`  // Max length for code snippet (prefix, or combined FIM parts)
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
	// Validate new fields
	if c.MaxPreambleLen <= 0 {
		log.Printf("Warning: max_preamble_len (%d) is not positive, using default %d", c.MaxPreambleLen, DefaultConfig.MaxPreambleLen)
		c.MaxPreambleLen = DefaultConfig.MaxPreambleLen
	}
	if c.MaxSnippetLen <= 0 {
		log.Printf("Warning: max_snippet_len (%d) is not positive, using default %d", c.MaxSnippetLen, DefaultConfig.MaxSnippetLen)
		c.MaxSnippetLen = DefaultConfig.MaxSnippetLen
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
	// --- Fields added in Step 8 ---
	MaxPreambleLen *int      `json:"max_preamble_len"`
	MaxSnippetLen  *int      `json:"max_snippet_len"`
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
	PromptPreamble     string // Built during analysis
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

// --- Cache Data Structure (Step 19/23) ---
// This struct holds the metadata and the serialized type information.
// It is encoded/decoded using gob for simplicity of handling the struct itself.
type CachedPackageData struct {
	SchemaVersion   int
	GoModHash       string
	InputFileHashes map[string]string // Key: relative path from package dir, Value: SHA256 hash
	TypesPackage    []byte            // Serialized using exportdata.Write
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
	// Specific error for cache issues
	ErrCache = errors.New("cache operation failed")
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
	Close() error                     // Added Step 24
	InvalidateCache(dir string) error // Added conceptually for Step 21/LSP
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
		// --- Defaults added in Step 8 ---
		MaxPreambleLen: 2048,
		MaxSnippetLen:  2048,
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
	var configParseError error // Track parse errors separately

	primaryPath, secondaryPath, pathErr := getConfigPaths()
	if pathErr != nil {
		loadErrors = append(loadErrors, pathErr)
		log.Printf("Warning: Could not determine config paths: %v", pathErr)
	}

	// Try loading from primary path first
	if primaryPath != "" {
		var loaded bool
		var loadErr error
		loaded, loadErr = loadAndMergeConfig(primaryPath, &cfg)
		if loadErr != nil {
			// Check if it was a parse error specifically
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				configParseError = loadErr // Store parse error
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", primaryPath, loadErr))
		}
		loadedFromFile = loaded // Mark if file existed, even if load failed
		if loadedFromFile && loadErr == nil {
			log.Printf("Loaded config from %s", primaryPath)
		} else if loadedFromFile && loadErr != nil {
			log.Printf("Attempted load from %s but failed.", primaryPath)
		}
	}

	// Try secondary path ONLY if primary was not found OR if primary existed but failed reading (not parsing)
	primaryExistedButFailedRead := len(loadErrors) > 0 && !errors.Is(loadErrors[len(loadErrors)-1], os.ErrNotExist) && configParseError == nil
	primaryNotFound := len(loadErrors) > 0 && errors.Is(loadErrors[len(loadErrors)-1], os.ErrNotExist)

	if !loadedFromFile && secondaryPath != "" && (primaryNotFound || primaryPath == "") {
		var loaded bool
		var loadErr error
		loaded, loadErr = loadAndMergeConfig(secondaryPath, &cfg)
		if loadErr != nil {
			if strings.Contains(loadErr.Error(), "parsing config file JSON") {
				// Prioritize parse error from primary path if both fail parsing
				if configParseError == nil { configParseError = loadErr }
			}
			loadErrors = append(loadErrors, fmt.Errorf("loading %s failed: %w", secondaryPath, loadErr))
		}
		loadedFromFile = loaded // Mark if file existed
		if loadedFromFile && loadErr == nil {
			log.Printf("Loaded config from %s", secondaryPath)
		} else if loadedFromFile && loadErr != nil {
			log.Printf("Attempted load from %s but failed.", secondaryPath)
		}
	}


	// Write default config if no file existed OR if parsing failed
	loadSucceeded := loadedFromFile && configParseError == nil

	if !loadSucceeded {
		if configParseError != nil {
			log.Printf("Existing config file failed to parse: %v. Attempting to write default.", configParseError)
		} else {
			log.Println("No config file found. Attempting to write default.")
		}

		writePath := primaryPath // Prefer primary path for writing
		if writePath == "" {
			writePath = secondaryPath // Fallback if primary failed determination
		}

		if writePath != "" {
			log.Printf("Attempting to write default config to %s", writePath)
			// Use validated DefaultConfig for writing
			if err := writeDefaultConfig(writePath, DefaultConfig); err != nil {
				log.Printf("Warning: Failed to write default config: %v", err)
				loadErrors = append(loadErrors, fmt.Errorf("writing default config failed: %w", err))
			}
		} else {
			log.Println("Warning: Cannot determine path to write default config.")
			loadErrors = append(loadErrors, errors.New("cannot determine default config path"))
		}
		// If load failed (parse or not found), use default config
		cfg = DefaultConfig
	}

	// Assign internal templates if not overridden (they shouldn't be via file)
	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = promptTemplate
	}
	if cfg.FimTemplate == "" {
		cfg.FimTemplate = fimPromptTemplate
	}

	// Return combined non-fatal errors wrapped in ErrConfig
	// Ensure final config is validated before returning
	finalCfg := cfg
	if err := finalCfg.Validate(); err != nil {
		 log.Printf("Warning: Config after load/merge failed validation: %v. Returning defaults.", err)
		 loadErrors = append(loadErrors, fmt.Errorf("post-load config validation failed: %w", err))
		 // Fallback to validated default config on error
		 if valErr := DefaultConfig.Validate(); valErr != nil {
			  return DefaultConfig, fmt.Errorf("default config is invalid: %w", valErr) // Should not happen
		  }
		 finalCfg = DefaultConfig
	}

	if len(loadErrors) > 0 {
		// Return the validated config (might be defaults) along with the load errors
		return finalCfg, fmt.Errorf("%w: %w", ErrConfig, errors.Join(loadErrors...))
	}

	return finalCfg, nil // Return validated config and nil error
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
		// Fallback path using HOME if XDG fails but HOME works
		if primary == "" && cfgErr != nil { // Only use fallback if XDG failed
			 primary = filepath.Join(homeDir, ".config", configDirName, defaultConfigFileName)
			 log.Printf("Using fallback primary config path: %s", primary)
		}
		secondary = filepath.Join(homeDir, ".local", "share", configDirName, defaultConfigFileName)
	} else {
		log.Printf("Warning: Could not determine user home directory: %v", homeErr)
	}

	// Report error only if BOTH primary and secondary path determination failed
	if primary == "" && secondary == "" {
		 // Combine potential errors if both lookups failed
		 err = fmt.Errorf("cannot determine config/home directories: config error: %v; home error: %v", cfgErr, homeErr)
	}
	return primary, secondary, err
}


// loadAndMergeConfig attempts to load config from a path and merge it into cfg.
// Returns true if file existed and was attempted to be loaded, false otherwise.
// Error indicates issues during read/parse.
func loadAndMergeConfig(path string, cfg *Config) (bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil // File doesn't exist, not an error here
		}
		// Indicate file existed but couldn't be read
		return true, fmt.Errorf("reading config file %q failed: %w", path, err)
	}
	// File exists
	if len(data) == 0 {
		log.Printf("Warning: Config file exists but is empty: %s", path)
		return true, nil // Treat as loaded but with no overrides
	}

	var fileCfg FileConfig
	if err := json.Unmarshal(data, &fileCfg); err != nil {
		// Don't treat parse error as "loaded successfully", return error but indicate file existed
		return true, fmt.Errorf("parsing config file JSON %q failed: %w", path, err)
	}

	// Merge fields if they exist in the file config
	if fileCfg.OllamaURL != nil { cfg.OllamaURL = *fileCfg.OllamaURL }
	if fileCfg.Model != nil { cfg.Model = *fileCfg.Model }
	if fileCfg.MaxTokens != nil { cfg.MaxTokens = *fileCfg.MaxTokens }
	if fileCfg.Stop != nil { cfg.Stop = *fileCfg.Stop }
	if fileCfg.Temperature != nil { cfg.Temperature = *fileCfg.Temperature }
	if fileCfg.UseAst != nil { cfg.UseAst = *fileCfg.UseAst }
	if fileCfg.UseFim != nil { cfg.UseFim = *fileCfg.UseFim }
	// Merge new fields (Step 8)
	if fileCfg.MaxPreambleLen != nil { cfg.MaxPreambleLen = *fileCfg.MaxPreambleLen }
	if fileCfg.MaxSnippetLen != nil { cfg.MaxSnippetLen = *fileCfg.MaxSnippetLen }

	return true, nil // File existed and was successfully parsed & merged
}


// writeDefaultConfig creates the directory and writes the default config as JSON.
func writeDefaultConfig(path string, defaultConfig Config) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0750); err != nil {
		return fmt.Errorf("failed to create config directory %s: %w", dir, err)
	}
	// Use structure matching FileConfig for exportable fields
	type ExportableConfig struct {
		OllamaURL      string   `json:"ollama_url"`
		Model          string   `json:"model"`
		MaxTokens      int      `json:"max_tokens"`
		Stop           []string `json:"stop"`
		Temperature    float64  `json:"temperature"`
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
		UseAst:         defaultConfig.UseAst,
		UseFim:         defaultConfig.UseFim,
		MaxPreambleLen: defaultConfig.MaxPreambleLen,
		MaxSnippetLen:  defaultConfig.MaxSnippetLen,
	}
	jsonData, err := json.MarshalIndent(expCfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal default config to JSON: %w", err)
	}
	// WriteFile ensures atomic write on most systems
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
	// Use a shared client with reasonable timeout
	return &httpOllamaClient{
		httpClient: &http.Client{Timeout: 90 * time.Second}, // Increased timeout slightly
	}
}

// GenerateStream handles the HTTP request to the Ollama generate endpoint.
func (c *httpOllamaClient) GenerateStream(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	// Ensure URL path joining is correct
	base := strings.TrimSuffix(config.OllamaURL, "/")
	endpointPath := "/api/generate"
	endpointURL := base + endpointPath // Simple concatenation works if base doesn't end with / and path starts with /

	u, err := url.Parse(endpointURL)
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL '%s': %w", endpointURL, err)
	}
	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"temperature": config.Temperature,
			"num_ctx":     4096, // Standard context size, maybe relate to prompt size later
			"top_p":       0.9,
			"stop":        config.Stop,
			// Use num_predict for max tokens based on Ollama API docs
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
	req.Header.Set("Accept", "application/x-ndjson") // Expect newline delimited JSON

	resp, err := c.httpClient.Do(req)
	if err != nil {
		// More specific error checking
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("%w: ollama request timed out after %v: %w", ErrOllamaUnavailable, c.httpClient.Timeout, err)
		}
		// Check for connection refused? net.OpError?
		var netErr *net.OpError
        if errors.As(err, &netErr) {
             if netErr.Op == "dial" {
                 return nil, fmt.Errorf("%w: connection refused or network error connecting to %s: %w", ErrOllamaUnavailable, u.Host, err)
             }
        }
		return nil, fmt.Errorf("%w: error making HTTP request: %w", ErrOllamaUnavailable, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(read error)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			// Try to parse Ollama's specific error format
			var ollamaErrResp struct { Error string `json:"error"` }
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error // Use specific error if available
			}
		}
		err = &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
		// Wrap specific OllamaError within ErrOllamaUnavailable
		return nil, fmt.Errorf("%w: %w", ErrOllamaUnavailable, err)
	}
	return resp.Body, nil // Return successful response body
}


// --- Code Analyzer (bbolt implementation - Step 23) ---

// GoPackagesAnalyzer implements Analyzer using go/packages and bbolt cache.
type GoPackagesAnalyzer struct {
	db *bbolt.DB    // DB handle added in Step 19/23
	mu sync.Mutex // Keep mutex for potential future concurrent Analyze calls
}

// NewGoPackagesAnalyzer creates a new Go code analyzer, opening the bbolt DB.
func NewGoPackagesAnalyzer() *GoPackagesAnalyzer {
	dbPath := ""
	userCacheDir, err := os.UserCacheDir()
	if err == nil {
		// Place DB itself directly in cache dir for easier management?
		// Use configDirName for consistency
		dbDir := filepath.Join(userCacheDir, configDirName, "bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err == nil {
			dbPath = filepath.Join(dbDir, "pkgcache.db") // Simpler DB filename
		} else {
			log.Printf("Warning: Could not create bbolt cache directory %s: %v", dbDir, err)
		}
	} else {
		log.Printf("Warning: Could not determine user cache directory: %v. Caching disabled.", err)
	}

	var db *bbolt.DB
	if dbPath != "" {
		opts := &bbolt.Options{
			Timeout: 1 * time.Second, // Prevent indefinite wait if db locked elsewhere
			// NoGrow: true, // Consider if DB file should preallocate or grow
		}
		db, err = bbolt.Open(dbPath, 0600, opts)
		if err != nil {
			log.Printf("Warning: Failed to open bbolt cache file %s: %v. Caching will be disabled.", dbPath, err)
			db = nil // Disable caching if open fails
		} else {
			// Ensure cache bucket exists
			err = db.Update(func(tx *bbolt.Tx) error {
				_, err := tx.CreateBucketIfNotExists(cacheBucketName)
				if err != nil {
					return fmt.Errorf("failed to create cache bucket %s: %w", string(cacheBucketName), err)
				}
				return nil
			})
			if err != nil {
				log.Printf("Warning: Failed to ensure bbolt bucket exists: %v. Caching disabled.", err)
				db.Close() // Close DB if bucket creation failed
				db = nil
			} else {
				log.Printf("Using bbolt cache at %s", dbPath)
			}
		}
	}

	return &GoPackagesAnalyzer{
		db: db,
	}
}

// Close closes the underlying bbolt database. (Added Step 24)
func (a *GoPackagesAnalyzer) Close() error {
	a.mu.Lock() // Lock mutex if closing could race with Analyze calls
	defer a.mu.Unlock()
	if a.db != nil {
		log.Println("Closing bbolt cache database.")
		err := a.db.Close()
		a.db = nil // Ensure handle is nil after closing
		return err
	}
	return nil
}

// Analyze parses the file, performs type checking using cache, and extracts context.
func (a *GoPackagesAnalyzer) Analyze(ctx context.Context, filename string, line, col int) (info *AstContextInfo, analysisErr error) {
	info = &AstContextInfo{
		FilePath:         filename, // Keep original filename initially
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0),
		CallArgIndex:     -1,
	}
	// Defer panic recovery
	defer func() {
		r := recover()
		if r != nil {
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			addAnalysisError(info, panicErr)
			// Ensure analysisErr reflects the panic if it's the only error
			if analysisErr == nil { analysisErr = panicErr } else { analysisErr = errors.Join(analysisErr, panicErr) }
			log.Printf("Panic recovered during AnalyzeCodeContext: %v\n%s", r, string(debug.Stack()))
		}
	}()

	absFilename, err := filepath.Abs(filename)
	if err != nil {
		// Cannot proceed without absolute path for directory context
		return info, fmt.Errorf("failed to get absolute path for '%s': %w", filename, err)
	}
	info.FilePath = absFilename // Use absolute path from now on
	log.Printf("Starting context analysis for: %s (%d:%d)", absFilename, line, col)
	dir := filepath.Dir(absFilename)

	// --- Cache Key Generation ---
	goModHash := calculateGoModHash(dir) // Helper calculates hash or placeholder
	cacheKey := []byte(dir + "::" + goModHash) // bbolt uses byte slices for keys
	// ---

	var targetPkg *packages.Package
	var targetFileAST *ast.File
	var targetFile *token.File
	// Errors associated with the current attempt (cache or load)
	var currentLoadErrors []error
	var loadDuration time.Duration // Time for load OR cache reconstruction
	var stepsDuration time.Duration // Time for performAnalysisSteps
	var preambleDuration time.Duration // Time for buildPreamble
	cacheHit := false
	var cachedMetaData *CachedPackageData // Hold decoded metadata (excluding TypesPackage initially)

	// --- Try Loading from Cache (bbolt View transaction) ---
	if a.db != nil {
		readStart := time.Now()
		dbErr := a.db.View(func(tx *bbolt.Tx) error {
			b := tx.Bucket(cacheBucketName)
			if b == nil { return fmt.Errorf("cache bucket %s not found during View", string(cacheBucketName)) }
			valBytes := b.Get(cacheKey)
			if valBytes == nil { return nil /* Cache miss - key not found */ }

			// Decode the metadata value using gob
			var decoded CachedPackageData
			decoder := gob.NewDecoder(bytes.NewReader(valBytes))
			if err := decoder.Decode(&decoded); err != nil {
				log.Printf("Warning: Failed to gob-decode cached data header for key %s: %v", string(cacheKey), err)
				return nil // Treat as miss
			}
			if decoded.SchemaVersion != cacheSchemaVersion {
				log.Printf("Warning: Cache data for key %s has old schema version %d. Ignoring.", string(cacheKey), decoded.SchemaVersion)
				return nil // Treat as miss
			}
			cachedMetaData = &decoded
			return nil
		})
		if dbErr != nil {
			log.Printf("Warning: Error reading from bbolt cache: %v", dbErr)
		}
		log.Printf("DEBUG: Cache read attempt took %v", time.Since(readStart))
	} else {
		 log.Printf("DEBUG: Cache disabled (db handle is nil).")
	}
	// --- End Cache Read Transaction ---


	// --- Validate and Reconstruct if potential hit ---
	if cachedMetaData != nil {
		validationStart := time.Now()
		log.Printf("DEBUG: Potential cache hit for key: %s. Validating...", string(cacheKey))
		// Pass nil for pkg hint, calculate based on dir scan
		currentHashes, hashErr := calculateInputHashes(dir, nil)
		// Check go.mod hash as well
		if hashErr == nil && cachedMetaData.GoModHash == goModHash && compareFileHashes(currentHashes, cachedMetaData.InputFileHashes) {
			log.Printf("DEBUG: Cache VALID for key: %s. Attempting to reconstruct package...", string(cacheKey))
			reconStart := time.Now()
			fset := token.NewFileSet() // New fileset needed for reconstruction

			// Decode types using exportdata.Read (Step 25)
			typesPkg, decodeErr := decodeTypesPackage(cachedMetaData.TypesPackage, fset, dir) // Pass fset and dir

			if decodeErr == nil && typesPkg != nil {
				// Re-parse ASTs using the same fileset as decoded types need
				// Pass context for potential cancellation during parse
				parsedFiles, parseErrs := parsePackageFiles(ctx, dir, fset)
				currentLoadErrors = append(currentLoadErrors, parseErrs...) // Add parse errors to current attempt's errors

				// Only proceed if parsing was reasonably successful (e.g., got some files)
				if len(parsedFiles) > 0 {
					// Reconstruct minimal packages.Package.
					// WARNING: TypeInfo (Defs, Uses) is not fully reconstructed here.
					targetPkg = &packages.Package{
						Name:    typesPkg.Name(), PkgPath: typesPkg.Path(), Fset:    fset,
						Syntax:  parsedFiles, Types:   typesPkg,
						TypesInfo: &types.Info{ // Provide empty maps as fallback
							Types: make(map[ast.Expr]types.TypeAndValue),
							Defs:  make(map[*ast.Ident]types.Object),
							Uses:  make(map[*ast.Ident]types.Object),
						},
						CompiledGoFiles: keysFromAstFileMap(parsedFiles, fset),
					}

					// TODO: Attempt to populate TypesInfo more accurately here after decoding types and parsing ASTs.

					// Find the specific target file within the reconstructed package
					foundFile := false
					for _, astF := range targetPkg.Syntax {
						if astF != nil {
							tokF := fset.File(astF.Pos())
							if tokF != nil && tokF.Name() != "" {
								normTokFName, _ := filepath.Abs(tokF.Name())
								if normTokFName == absFilename {
									targetFileAST = astF; targetFile = tokF; foundFile = true; break
								}
							}
						}
					}

					if foundFile {
						cacheHit = true
						loadDuration = time.Since(reconStart) // Record reconstruction time
						log.Printf("DEBUG: Package reconstructed from cache in %v (TypesInfo may be limited)", loadDuration)
					} else {
						log.Printf("Warning: Failed to find target file %s in reconstructed package. Cache invalid.", absFilename)
						targetPkg = nil // Reset
						if a.db != nil { a.deleteCacheEntry(cacheKey) } // Invalidate
						currentLoadErrors = append(currentLoadErrors, fmt.Errorf("failed to find file %s in reconstructed package", absFilename))
					}
				} else { // Parsing failed badly
					 log.Printf("Warning: Failed to parse any files on cache hit for %s. Cache invalid.", dir)
					 targetPkg = nil // Reset
					 if a.db != nil { a.deleteCacheEntry(cacheKey) } // Invalidate cache
					 currentLoadErrors = append(currentLoadErrors, fmt.Errorf("failed to parse files on cache hit for %s", dir))
				}
			} else {
				log.Printf("Warning: Failed to decode types package from cache: %v. Treating as miss.", decodeErr)
				if a.db != nil && decodeErr != nil { a.deleteCacheEntry(cacheKey) } // Delete potentially corrupted entry
				currentLoadErrors = append(currentLoadErrors, decodeErr) // Record decode error
			}
		} else {
			log.Printf("DEBUG: Cache INVALID for key: %s (HashErr: %v, Hash Match: %t, GoMod Match: %t). Treating as miss.",
				string(cacheKey), hashErr, compareFileHashes(currentHashes, cachedMetaData.InputFileHashes), cachedMetaData.GoModHash == goModHash)
			// Delete the invalid cache file
			if a.db != nil && cachedMetaData != nil {
				 a.deleteCacheEntry(cacheKey)
			}
			if hashErr != nil { currentLoadErrors = append(currentLoadErrors, hashErr)} // Record hash error
		}
		log.Printf("DEBUG: Cache validation/reconstruction took %v", time.Since(validationStart))
		// Add any reconstruction errors to the main analysis errors
		for _, reconErr := range currentLoadErrors { addAnalysisError(info, reconErr) }
	}
	// --- End Validation ---


	// --- Load if Cache Miss ---
	if !cacheHit {
		// If cache is disabled or miss/invalid, perform full load
		if a.db == nil { log.Printf("DEBUG: Cache disabled, loading via packages.Load...") } else { log.Printf("DEBUG: Cache miss or invalid for key: %s. Loading via packages.Load...", string(cacheKey)) }

		loadStart := time.Now()
		fset := token.NewFileSet() // packages.Load needs its own fileset
		// Reset current load errors before calling loadPackageInfo
		currentLoadErrors = nil
		// Perform actual package loading
		targetPkg, targetFileAST, targetFile, currentLoadErrors = loadPackageInfo(ctx, absFilename, fset)
		loadDuration = time.Since(loadStart)
		log.Printf("DEBUG: packages.Load completed in %v", loadDuration)
		// Add errors from this load attempt to the main analysis errors
		for _, loadErr := range currentLoadErrors { addAnalysisError(info, loadErr) }


		// Save to cache on successful load (check currentLoadErrs specifically)
		if a.db != nil && targetPkg != nil && targetPkg.Types != nil && len(currentLoadErrors) == 0 {
			log.Printf("DEBUG: Attempting to save package to bbolt cache. Key: %s", string(cacheKey))
			saveStart := time.Now()
			// Pass loaded targetPkg to get accurate file list
			inputHashes, hashErr := calculateInputHashes(dir, targetPkg)
			if hashErr == nil {
				// Encode types using exportdata.Write (Step 25)
				typesData, marshalErr := encodeTypesPackage(targetPkg.Fset, targetPkg.Types) // Pass fset and pkg.Types
				if marshalErr == nil {
					cacheDataToSave := CachedPackageData{
						SchemaVersion:   cacheSchemaVersion,
						GoModHash:       goModHash,
						InputFileHashes: inputHashes,
						TypesPackage:    typesData,
					}
					// Encode cache struct using gob (only for the metadata + types bytes)
					var buf bytes.Buffer
					encoder := gob.NewEncoder(&buf)
					if encodeErr := encoder.Encode(&cacheDataToSave); encodeErr == nil {
						encodedBytes := buf.Bytes()
						// Save using bbolt Update transaction
						saveErr := a.db.Update(func(tx *bbolt.Tx) error {
							b := tx.Bucket(cacheBucketName)
							if b == nil { return fmt.Errorf("cache bucket %s disappeared", string(cacheBucketName)) }
							log.Printf("DEBUG: Writing %d bytes to cache for key %s", len(encodedBytes), string(cacheKey))
							return b.Put(cacheKey, encodedBytes)
						})
						if saveErr == nil {
							log.Printf("DEBUG: Saved package to bbolt cache %s in %v", string(cacheKey), time.Since(saveStart))
						} else {
							log.Printf("Warning: Failed to write to bbolt cache for key %s: %v", string(cacheKey), saveErr)
						}
					} else {
						log.Printf("Warning: Failed to gob-encode cache metadata for key %s: %v", string(cacheKey), encodeErr)
					}
				} else {
					log.Printf("Warning: Failed to encode types package using exportdata for caching: %v", marshalErr)
				}
			} else {
				log.Printf("Warning: Failed to calculate input hashes for caching: %v", hashErr)
			}
		} else if a.db != nil {
			// Log why saving didn't happen if load wasn't successful
			 log.Printf("DEBUG: Skipping cache save for key %s due to load errors (%d) or missing types.", string(cacheKey), len(currentLoadErrors))
		}
	}
	// --- End Load ---


	// --- Perform Analysis Steps ---
	var currentFset *token.FileSet
	if targetPkg != nil && targetPkg.Fset != nil {
		 currentFset = targetPkg.Fset // Use fset from loaded/reconstructed package
	} else if targetFile != nil {
		 // If targetFile exists, it MUST have been added to a fileset.
		 // Attempt to get FileSet from targetFile if targetPkg was nil/incomplete
		 // This is brittle. LoadPackageInfo / reconstruction should guarantee Fset if file exists.
		 log.Printf("Error: targetFile exists but associated FileSet is missing. Cannot perform analysis steps.")
		 addAnalysisError(info, errors.New("internal error: FileSet missing for analysis steps"))
		 targetFile = nil // Prevent calling steps without fset
	}


	if targetFile != nil && currentFset != nil { // Check fset explicitly
		stepsStart := time.Now()
		// Pass potentially reconstructed targetPkg and the correct fset
		analyzeStepErr := a.performAnalysisSteps(targetFile, targetFileAST, targetPkg, currentFset, line, col, info)
		stepsDuration = time.Since(stepsStart)
		log.Printf("DEBUG: performAnalysisSteps completed in %v", stepsDuration)
		if analyzeStepErr != nil { addAnalysisError(info, analyzeStepErr) }
	} else {
		// Log if steps skipped due to missing components
		if targetFile == nil { addAnalysisError(info, errors.New("cannot perform analysis steps: missing target file after load/cache check"))}
		if currentFset == nil && targetFile != nil { addAnalysisError(info, errors.New("cannot perform analysis steps: missing FileSet after load/cache check"))}
	}
	// --- End Analysis Steps ---


	// --- Build Preamble ---
	var qualifier types.Qualifier
	if targetPkg != nil && targetPkg.Types != nil {
		qualifier = types.RelativeTo(targetPkg.Types)
	} else {
		// Fallback qualifier if types are missing (e.g., direct parse, bad cache decode)
		qualifier = func(other *types.Package) string {
			if other != nil { return other.Path() } // Use path if available
			return "" // Otherwise, empty string (no qualification)
		}
		// Log if using fallback due to missing types
		if targetPkg == nil || targetPkg.Types == nil {
			 log.Printf("DEBUG: Building preamble with limited/no type info.")
		}
	}
	preambleStart := time.Now()
	info.PromptPreamble = buildPreamble(info, qualifier) // Pass qualifier, not package
	preambleDuration = time.Since(preambleStart)
	log.Printf("DEBUG: buildPreamble completed in %v", preambleDuration)
	// Limit preamble logging if too long
	if len(info.PromptPreamble) > 500 {
		 log.Printf("Generated Context Preamble (length %d):\n---\n%s\n... (preamble truncated in log)\n---", len(info.PromptPreamble), info.PromptPreamble[:500])
	} else {
		 log.Printf("Generated Context Preamble (length %d):\n---\n%s\n---", len(info.PromptPreamble), info.PromptPreamble)
	}
	// --- End Build Preamble ---


	logAnalysisErrors(info.AnalysisErrors)
	analysisErr = errors.Join(info.AnalysisErrors...) // Combine non-fatal errors

	log.Printf("Context analysis finished (Load/Recon: %v, Steps: %v, Preamble: %v, Cache Hit: %t)",
		loadDuration, stepsDuration, preambleDuration, cacheHit)


	// Wrap the combined analysis errors only if analysis produced NO usable result
	// Consider analysis failed if we couldn't even get the targetFile token
	if len(info.AnalysisErrors) > 0 && targetFile == nil {
		// If we have errors AND no cache hit AND no target file, treat as fatal analysis failure
		 return info, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisErr)
	 } else if len(info.AnalysisErrors) > 0 {
		 // Return info but also non-fatal error wrapper if some result was produced
		 return info, fmt.Errorf("%w: %w", ErrAnalysisFailed, analysisErr)
	 }


	return info, nil // Success or partial success with non-fatal errors handled
}


// --- InvalidateCache Method (Placeholder for LSP handlers - Step 21) ---
// Simple invalidation by deleting the specific key.
func (a *GoPackagesAnalyzer) InvalidateCache(dir string) error {
	a.mu.Lock() // Lock if db handle might be checked/closed concurrently
	db := a.db
	a.mu.Unlock()

	if db == nil {
		 log.Printf("DEBUG: Cache invalidation skipped: DB is nil.")
		 return nil
	 } // Cache disabled or closed

	goModHash := calculateGoModHash(dir)
	cacheKey := []byte(dir + "::" + goModHash)

	log.Printf("DEBUG: Invalidating cache for key: %s", string(cacheKey))
	err := db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(cacheBucketName)
		if b == nil {
			 log.Printf("Warning: Cache bucket %s not found during invalidation.", string(cacheBucketName))
			 return nil // Bucket doesn't exist, nothing to delete
		 }
		existing := b.Get(cacheKey)
		if existing == nil {
			 log.Printf("DEBUG: Cache entry for key %s already deleted or never existed.", string(cacheKey))
			 return nil // Key doesn't exist, nothing to delete
		 }
		log.Printf("DEBUG: Deleting cache entry for key: %s", string(cacheKey))
		return b.Delete(cacheKey)
	})
	if err != nil {
		log.Printf("Warning: Failed to delete cache entry %s: %v", string(cacheKey), err)
		return fmt.Errorf("failed to delete cache entry %s: %w", string(cacheKey), err) // Return error
	}
	log.Printf("DEBUG: Successfully invalidated cache entry for key: %s", string(cacheKey))
	return nil
}



// performAnalysisSteps encapsulates the core analysis logic after loading/parsing.
// Takes fset as argument now.
func (a *GoPackagesAnalyzer) performAnalysisSteps(
	targetFile *token.File,
	targetFileAST *ast.File,
	targetPkg *packages.Package, // Can be nil/partial if reconstructed from cache
	fset *token.FileSet,         // Use the fileset associated with targetPkg/targetFile
	line, col int,
	info *AstContextInfo,
) error {
	if targetFile == nil || fset == nil {
		 return errors.New("performAnalysisSteps requires non-nil targetFile and fset")
	}
	// This function now assumes targetFile and fset are not nil
	cursorPos, posErr := calculateCursorPos(targetFile, line, col)
	if posErr != nil {
		// Don't addAnalysisError here, return directly as it's fatal for subsequent steps
		return fmt.Errorf("cursor position error: %w", posErr)
	}
	info.CursorPos = cursorPos
	// Use PositionFor for potentially more detailed position info if needed, includes filename
	log.Printf("Calculated cursor token.Pos: %d (%s)", info.CursorPos, fset.PositionFor(info.CursorPos, true))


	// AST can be nil if only types were cached and re-parse failed
	if targetFileAST != nil {
		 path := findEnclosingPath(targetFileAST, info.CursorPos, info) // Can return nil path
		 // Pass fset for logging positions accurately within findContextNodes
		 findContextNodes(path, info.CursorPos, targetPkg, fset, info)
		 // Pass fset for formatting receiver type accurately within gatherScopeContext
		 gatherScopeContext(path, targetPkg, fset, info)
		 findRelevantComments(targetFileAST, path, info.CursorPos, fset, info)
	} else {
		 addAnalysisError(info, errors.New("cannot perform detailed AST analysis: targetFileAST is nil (cache hit?)"))
		 // Still attempt to gather package scope if types available (targetPkg might exist from cache)
		 gatherScopeContext(nil, targetPkg, fset, info) // Pass nil path
	}
	return nil // Non-fatal errors are added to info.AnalysisErrors
}


// --- Prompt Formatter ---

// templateFormatter implements PromptFormatter using standard templates.
type templateFormatter struct{}

// newTemplateFormatter creates a new prompt formatter. (unexported)
func newTemplateFormatter() *templateFormatter {
	return &templateFormatter{}
}

// FormatPrompt generates the final prompt string, including truncation (Step 6/Step 8)
func (f *templateFormatter) FormatPrompt(contextPreamble string, snippetCtx SnippetContext, config Config) string {
	var finalPrompt string
	template := config.PromptTemplate

	// Use configured limits (Step 8)
	maxPreambleLen := config.MaxPreambleLen
	maxSnippetLen := config.MaxSnippetLen
	maxFIMPartLen := maxSnippetLen / 2 // Split budget for FIM

	// Truncation Logic (Simple version kept from Step 6/8)
	if len(contextPreamble) > maxPreambleLen {
		log.Printf("Warning: Truncating context preamble from %d to %d bytes.", len(contextPreamble), maxPreambleLen)
		// Truncate from the beginning, keeping the end
		startByte := len(contextPreamble) - maxPreambleLen + len("... (context truncated)\n")
		if startByte < 0 { startByte = 0 } // Avoid negative index if limit is very small
		contextPreamble = "... (context truncated)\n" + contextPreamble[startByte:]
	}

	if config.UseFim {
		template = config.FimTemplate
		prefix := snippetCtx.Prefix
		suffix := snippetCtx.Suffix

		// Truncate FIM parts individually
		if len(prefix) > maxFIMPartLen {
			log.Printf("Warning: Truncating FIM prefix from %d to %d bytes.", len(prefix), maxFIMPartLen)
			// Truncate from beginning
			startByte := len(prefix) - maxFIMPartLen + len("...(prefix truncated)")
			if startByte < 0 { startByte = 0 }
			prefix = "...(prefix truncated)" + prefix[startByte:]
		}
		if len(suffix) > maxFIMPartLen {
			log.Printf("Warning: Truncating FIM suffix from %d to %d bytes.", len(suffix), maxFIMPartLen)
			// Truncate from end
			endByte := maxFIMPartLen - len("(suffix truncated)...")
			if endByte < 0 { endByte = 0 }
			suffix = suffix[:endByte] + "(suffix truncated)..."
		}

		// Assemble the prompt using the FIM template structure
		// Assuming template has %s placeholders for CONTEXT, PRE, SUF
		finalPrompt = fmt.Sprintf(template, contextPreamble, prefix, suffix)

	} else { // Standard completion
		snippet := snippetCtx.Prefix
		// Truncate snippet (keep the end)
		if len(snippet) > maxSnippetLen {
			log.Printf("Warning: Truncating code snippet from %d to %d bytes.", len(snippet), maxSnippetLen)
			// Truncate from beginning
			startByte := len(snippet) - maxSnippetLen + len("...(code truncated)\n")
			if startByte < 0 { startByte = 0 }
			snippet = "...(code truncated)\n" + snippet[startByte:]
		}
		// Format with potentially truncated parts
		// Assuming template placeholders: CONTEXT, CODE
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
	analyzer  Analyzer // Interface now
	formatter PromptFormatter
	config    Config
}

// NewDeepCompleter creates a new DeepCompleter service with default components. Exported.
func NewDeepCompleter() (*DeepCompleter, error) {
	cfg, configErr := LoadConfig()
	// Handle fatal vs non-fatal config errors
	if configErr != nil && !errors.Is(configErr, ErrConfig) { // Fatal error getting paths etc.
		return nil, configErr
	} else if configErr != nil { // Non-fatal load/write error
		log.Printf("Warning during config load: %v", configErr)
		// Proceed with default config even if loading/writing failed non-fatally
		// Ensure cfg is valid before proceeding
		if err := cfg.Validate(); err != nil {
			 log.Printf("Warning: Config after failed load/merge is invalid: %v. Using pure defaults.", err)
			 cfg = DefaultConfig // Use pure defaults if loaded one is bad
			 // Validate defaults just in case
			 if valErr := cfg.Validate(); valErr != nil {
				 return nil, fmt.Errorf("default config validation failed: %w", valErr)
			 }
		}
	} else {
		// If no error during load, still validate the loaded config
		 if err := cfg.Validate(); err != nil {
			  log.Printf("Warning: Loaded config validation failed: %v. Falling back to defaults.", err)
			  // Fallback to validated default config on error
			  if valErr := DefaultConfig.Validate(); valErr != nil {
				   return nil, fmt.Errorf("default config validation also failed: %w", valErr)
			   }
			  cfg = DefaultConfig
			  // Optionally, wrap the validation error to inform the caller?
			  // configErr = fmt.Errorf("%w: loaded config validation: %w", ErrInvalidConfig, err)
		  }
	}

	// At this point, cfg should be a validated config (either loaded or default)
	analyzer := NewGoPackagesAnalyzer() // Create bbolt analyzer

	// Return potential non-fatal config load error along with completer
	dc := &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer, // Use the new analyzer
		formatter: newTemplateFormatter(),
		config:    cfg,
	}
	// Return non-fatal errors from LoadConfig if any occurred
	if configErr != nil && errors.Is(configErr, ErrConfig) {
		return dc, configErr
	}
	return dc, nil
}


// NewDeepCompleterWithConfig creates a new DeepCompleter service with provided config. Exported.
func NewDeepCompleterWithConfig(config Config) (*DeepCompleter, error) {
	// Ensure internal templates are set if empty (can happen if called directly)
	if config.PromptTemplate == "" { config.PromptTemplate = promptTemplate }
	if config.FimTemplate == "" { config.FimTemplate = fimPromptTemplate }

	// Validate provided config
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}

	analyzer := NewGoPackagesAnalyzer() // Create bbolt analyzer

	return &DeepCompleter{
		client:    newHttpOllamaClient(),
		analyzer:  analyzer,
		formatter: newTemplateFormatter(),
		config:    config,
	}, nil
}

// Close cleans up resources used by DeepCompleter, like the analyzer cache. (Added Step 24)
func (dc *DeepCompleter) Close() error {
	if dc.analyzer != nil {
		return dc.analyzer.Close()
	}
	return nil
}


// GetCompletion provides completion for a raw code snippet. Non-streaming. Exported.
func (dc *DeepCompleter) GetCompletion(ctx context.Context, codeSnippet string) (string, error) {
	log.Println("DeepCompleter.GetCompletion called for basic prompt.")
	// For raw snippets, AST analysis isn't applicable.
	contextPreamble := "// Provide Go code completion below."
	snippetCtx := SnippetContext{Prefix: codeSnippet}

	// Use the formatter with the current config (which includes length limits)
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, dc.config)
	// Avoid logging potentially large raw snippets in production/normal use
	// log.Printf("Generated Prompt (length %d):\n---\n%s\n---", len(prompt), prompt)


	reader, err := dc.client.GenerateStream(ctx, prompt, dc.config)
	if err != nil {
		// Simplify error checking: if it wraps ErrOllamaUnavailable, return that.
		if errors.Is(err, ErrOllamaUnavailable) {
			 return "", err // Return wrapped error directly
		}
		// Otherwise, wrap generic API call failure
		return "", fmt.Errorf("failed to call Ollama API for basic prompt: %w", err)
	}
	var buffer bytes.Buffer
	streamCtx, cancelStream := context.WithTimeout(ctx, 50*time.Second) // Use reasonable timeout
	defer cancelStream()
	if streamErr := streamCompletion(streamCtx, reader, &buffer); streamErr != nil {
		// Check if error is context related (e.g. timeout)
		if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
			 return "", fmt.Errorf("%w: streaming context error: %w", ErrOllamaUnavailable, streamErr)
		 }
		return "", fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
	}
	// Trim whitespace from the final completed buffer
	return strings.TrimSpace(buffer.String()), nil
}


// GetCompletionStreamFromFile uses AST context and streams completion. Exported.
func (dc *DeepCompleter) GetCompletionStreamFromFile(ctx context.Context, filename string, row, col int, w io.Writer) error {
	var contextPreamble string = "// Basic file context only."
	var analysisErr error // Stores combined non-fatal errors from analysis
	var analysisInfo *AstContextInfo // Hold result from Analyze

	if dc.config.UseAst {
		log.Printf("Analyzing context using AST/Types for %s:%d:%d", filename, row, col)
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 30*time.Second) // Adjusted timeout
		// Call analyzer via interface
		analysisInfo, analysisErr = dc.analyzer.Analyze(analysisCtx, filename, row, col)
		cancelAnalysis() // Release context resources promptly

		// Handle fatal analysis errors immediately (errors not wrapping ErrAnalysisFailed)
		if analysisErr != nil && !errors.Is(analysisErr, ErrAnalysisFailed) {
			return fmt.Errorf("analysis failed fatally: %w", analysisErr)
		}
		// Log non-fatal errors (if analysisErr wraps ErrAnalysisFailed)
		// analysisInfo might still contain partial data even if analysisErr is non-nil
		if analysisErr != nil && analysisInfo != nil {
			// Log combined error details if present
			logAnalysisErrors(analysisInfo.AnalysisErrors)
		}

		// Build preamble based on results
		if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble
		} else if analysisErr != nil {
			// Include error summary in preamble if analysis failed non-fatally but produced no preamble
			contextPreamble += fmt.Sprintf("// Warning: AST analysis completed with errors and no context: %v\n", analysisErr)
		} else {
			 contextPreamble += "// Warning: AST analysis returned no specific context (or analysis failed silently).\n"
		}
	} else {
		log.Println("AST analysis disabled.")
	}

	snippetCtx, err := extractSnippetContext(filename, row, col)
	if err != nil {
		return fmt.Errorf("failed to extract code snippet context: %w", err)
	}

	// Formatter now handles truncation based on config
	prompt := dc.formatter.FormatPrompt(contextPreamble, snippetCtx, dc.config)
	// Limit logging of potentially large prompts
	if len(prompt) > 1000 {
		log.Printf("Generated Prompt (length %d):\n---\n%s\n... (prompt truncated in log)\n---", len(prompt), prompt[:1000])
	} else {
		log.Printf("Generated Prompt (length %d):\n---\n%s\n---", len(prompt), prompt)
	}

	// Retryable API call function
	apiCallFunc := func() error {
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second) // API call timeout
		defer cancelApi()
		reader, apiErr := dc.client.GenerateStream(apiCtx, prompt, dc.config)
		if apiErr != nil {
			// Check if retryable before wrapping
			var oe *OllamaError
			isRetryable := errors.As(apiErr, &oe) && (oe.Status == http.StatusServiceUnavailable || oe.Status == http.StatusTooManyRequests)
			isRetryable = isRetryable || errors.Is(apiErr, context.DeadlineExceeded) // Retry timeouts? Yes.

			if isRetryable { return apiErr } // Return raw error for retry logic

			// Wrap non-retryable errors clearly
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, apiErr)
		}
		// Stream output directly
		PrettyPrint(ColorGreen, "Completion:\n") // Print header before streaming
		streamErr := streamCompletion(apiCtx, reader, w)
		// Add newline after stream completes or errors (ensures prompt return)
		fmt.Fprintln(w)
		if streamErr != nil {
			// Check if stream error is retryable (e.g., context cancelled due to timeout)
			if errors.Is(streamErr, context.DeadlineExceeded) || errors.Is(streamErr, context.Canceled) {
				 return streamErr // Allow retry on timeout during stream
			 }
			// Wrap non-retryable stream errors
			return fmt.Errorf("%w: %w", ErrStreamProcessing, streamErr)
		}
		log.Println("Completion stream finished successfully for this attempt.")
		return nil
	}

	// Execute with retry logic
	err = retry(ctx, apiCallFunc, maxRetries, retryDelay)
	if err != nil {
		// Wrap final error after retries appropriately
		var oe *OllamaError
		// Simplify wrapping: check common error types
		if errors.As(err, &oe) || errors.Is(err, ErrOllamaUnavailable) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			return fmt.Errorf("%w: %w", ErrOllamaUnavailable, err) // Group common comms/timeout errors
		}
		if errors.Is(err, ErrStreamProcessing){
			 return err // Return stream processing error directly if it wasn't context related
		 }
		// Generic fallback wrap for unexpected errors after retry
		return fmt.Errorf("failed to get completion stream after retries: %w", err)
	}

	// Log analysis errors again as a final warning, even if completion succeeded
	if analysisErr != nil { // Contains non-fatal errors wrapped with ErrAnalysisFailed
		log.Printf("Warning: Completion succeeded, but context analysis encountered non-fatal errors: %v", analysisErr)
	}
	return nil // Success
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
			return ctx.Err() // Return context error directly
		default:
		}
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if len(line) > 0 { // Process final fragment before EOF
					if procErr := processLine(line, w); procErr != nil {
						log.Printf("Error processing final line before EOF: %v", procErr)
						return procErr // Return error if final process fails
					}
				}
				return nil // Successful end of stream
			}
			// Check if the error is due to context cancellation
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				// Otherwise, it's a read error
				log.Printf("Error reading from Ollama stream: %v", err)
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			}
		}
		// Process line normally
		if procErr := processLine(line, w); procErr != nil {
			log.Printf("Error processing Ollama response line: %v", procErr)
			return procErr // Return error if processing fails
		}
	}
}
func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil
	}
	var resp OllamaResponse
	// Use json.Unmarshal for better error reporting if needed
	if err := json.Unmarshal(line, &resp); err != nil {
		// Tolerate simple marshalling errors, maybe log differently?
		log.Printf("Debug: Ignoring non-JSON line from Ollama stream: %s", string(line))
		return nil // Continue processing next line
	}
	if resp.Error != "" {
		log.Printf("Ollama reported an error in stream: %s", resp.Error)
		// Consider if this should be returned as ErrStreamProcessing
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	}
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		log.Printf("Error writing completion chunk to output: %v", err)
		// Check if writer is closed (e.g. pipe broken)
		// This might indicate context cancellation too
		return fmt.Errorf("error writing to output: %w", err)
	}
	return nil
}


// --- Retry Helper ---
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration) error {
	var lastErr error
	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		select {
		case <-ctx.Done(): // Check context before even trying the operation
			log.Printf("Context cancelled before attempt %d: %v", i+1, ctx.Err())
			return ctx.Err()
		default:
		}

		lastErr = operation()
		if lastErr == nil {
			return nil // Success
		}

		// Check for context errors first (most likely signal to stop retrying)
		if errors.Is(lastErr, context.Canceled) || errors.Is(lastErr, context.DeadlineExceeded) {
			 log.Printf("Attempt %d failed due to context error: %v. Not retrying.", i+1, lastErr)
			 return lastErr // Don't retry on context errors
		}

		// Check for specific retryable Ollama errors
		var ollamaErr *OllamaError
		isRetryable := errors.As(lastErr, &ollamaErr) &&
			(ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)

		// Also consider general unavailability errors as potentially retryable
		isRetryable = isRetryable || errors.Is(lastErr, ErrOllamaUnavailable)

		if !isRetryable {
			log.Printf("Attempt %d failed with non-retryable error: %v", i+1, lastErr)
			return lastErr // Return non-retryable errors immediately
		}

		// If retryable, wait and log
		waitDuration := currentDelay
		// Add jitter maybe? time.Duration(rand.Intn(100)) * time.Millisecond
		log.Printf("Attempt %d failed with retryable error: %v. Retrying in %v...", i+1, lastErr, waitDuration)

		select {
		case <-ctx.Done():
			log.Printf("Context cancelled during retry wait: %v", ctx.Err())
			return ctx.Err() // Respect context cancellation during wait
		case <-time.After(waitDuration):
			// Optional: Implement exponential backoff
			// currentDelay *= 2
			// if currentDelay > maxDelay { currentDelay = maxDelay }
		}
	}
	log.Printf("Operation failed after %d retries.", maxRetries)
	// Return the last error encountered
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr)
}


// --- Analysis Helpers ---
// loadPackageInfo loads package data using go/packages.
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
			if err != nil { log.Printf("Parser error in %s (proceeding with partial AST): %v", filename, err) }
			return file, nil
		},
		Tests: true,
	}

	pkgs, loadErr := packages.Load(cfg, ".")
	if loadErr != nil {
		log.Printf("Error loading packages for %s: %v", dir, loadErr)
		loadErrors = append(loadErrors, fmt.Errorf("package loading failed: %w", loadErr))
	}

	if len(pkgs) > 0 {
		 packages.Visit(pkgs, nil, func(pkg *packages.Package) {
			 for _, err := range pkg.Errors {
				  errMsg := err.Error(); found := false
				  for _, existing := range loadErrors { if existing.Error() == errMsg { found = true; break } }
				  if !found { loadErrors = append(loadErrors, err) }
			  }
		  })
		 if len(loadErrors) > 0 { log.Printf("Detailed errors encountered during package loading for %s", dir) }
	} else if loadErr == nil {
		 loadErrors = append(loadErrors, fmt.Errorf("no packages found in directory %s", dir))
	}

	for _, pkg := range pkgs {
		if pkg == nil || pkg.Fset == nil || pkg.Syntax == nil {
			if pkg != nil { log.Printf("DEBUG: Skipping partially loaded package (ID: %s) due to missing Fset/Syntax", pkg.ID) } else { log.Printf("DEBUG: Skipping nil package in results for %s", dir) }
			continue
		}
		for i, syntaxFile := range pkg.Syntax {
			if syntaxFile == nil { continue }
			tokenFile := pkg.Fset.File(syntaxFile.Pos())
			if tokenFile != nil && tokenFile.Name() != "" {
				normSyntaxFileName, _ := filepath.Abs(tokenFile.Name())
				if normSyntaxFileName == absFilename {
					if pkg.TypesInfo == nil {
						 log.Printf("Warning: Type info missing for target package %s", pkg.PkgPath)
						 loadErrors = append(loadErrors, fmt.Errorf("type info missing for package %s", pkg.PkgPath))
					 }
					if pkg.Types == nil {
						 log.Printf("Warning: Types definition missing for target package %s", pkg.PkgPath)
						 loadErrors = append(loadErrors, fmt.Errorf("types definition missing for package %s", pkg.PkgPath))
					 }
					return pkg, syntaxFile, tokenFile, loadErrors
				}
			} else if tokenFile == nil { log.Printf("DEBUG: Skipping syntax file in package %s - could not get token.File at Pos %v", pkg.ID, syntaxFile.Pos()) } else { log.Printf("DEBUG: Skipping syntax file in package %s - token.File Name is empty", pkg.ID) }
		}
	}

	if len(loadErrors) == 0 {
		 loadErrors = append(loadErrors, fmt.Errorf("target file %s not found within loaded packages for directory %s", absFilename, dir))
	}
	return nil, nil, nil, loadErrors
}

// directParse attempts to parse a single file directly if package loading fails.
func directParse(absFilename string, fset *token.FileSet) (*ast.File, *token.File, error) {
	srcBytes, readErr := os.ReadFile(absFilename)
	if readErr != nil {
		return nil, nil, fmt.Errorf("direct read failed for '%s': %w", absFilename, readErr)
	}
	const mode = parser.ParseComments | parser.AllErrors
	targetFileAST, parseErr := parser.ParseFile(fset, absFilename, srcBytes, mode)
	if parseErr != nil { log.Printf("Direct parsing of %s encountered errors (proceeding with partial AST): %v", absFilename, parseErr) }

	if targetFileAST == nil {
		finalErr := parseErr; if finalErr == nil { finalErr = errors.New("unknown error obtaining AST")}
		return nil, nil, fmt.Errorf("failed to obtain any AST from direct parse: %w", finalErr)
	}
	targetFile := fset.File(targetFileAST.Pos())
	if targetFile == nil {
		 return targetFileAST, nil, errors.New("failed to get token.File from fileset after direct parse")
	}
	return targetFileAST, targetFile, parseErr
}


// findEnclosingPath finds the AST path from root to the node enclosing the cursor.
func findEnclosingPath(targetFileAST *ast.File, cursorPos token.Pos, info *AstContextInfo) []ast.Node {
	if targetFileAST == nil { addAnalysisError(info, errors.New("cannot find enclosing path: targetFileAST is nil")); return nil }
	if !cursorPos.IsValid() { addAnalysisError(info, errors.New("cannot find enclosing path: invalid cursor position")); return nil }
	path, _ := astutil.PathEnclosingInterval(targetFileAST, cursorPos, cursorPos)
	if path == nil { log.Printf("DEBUG: No AST path found enclosing cursor position %v", cursorPos) }
	return path
}


// gatherScopeContext walks up the AST path to find enclosing function/block and gather scope variables.
// Takes fset as argument for receiver formatting.
func gatherScopeContext(path []ast.Node, targetPkg *packages.Package, fset *token.FileSet, info *AstContextInfo) {
	if fset == nil && path != nil { log.Println("Warning: Cannot format receiver type - fset is nil in gatherScopeContext.") }

	var qualifier types.Qualifier
	if targetPkg != nil && targetPkg.Types != nil {
		qualifier = types.RelativeTo(targetPkg.Types)
	} else {
		qualifier = func(other *types.Package) string { if other != nil { return other.Path() }; return "" }
	}

	if path != nil {
		for i := len(path) - 1; i >= 0; i-- {
			node := path[i]
			switch n := node.(type) {
			case *ast.FuncDecl:
				if info.EnclosingFuncNode == nil {
					info.EnclosingFuncNode = n
					if fset != nil && n.Recv != nil && len(n.Recv.List) > 0 && n.Recv.List[0].Type != nil {
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
							if info.EnclosingFunc == nil {
								 info.EnclosingFunc = fn
								 if sig, ok := fn.Type().(*types.Signature); ok {
									 addSignatureToScope(sig, info.VariablesInScope)
								 } else { log.Printf("Warning: Could not assert *types.Signature for func %s", n.Name.Name) }
							}
						}
					} else {
						 if info.EnclosingFunc == nil && n.Name != nil {
							  addAnalysisError(info, fmt.Errorf("definition for func '%s' not found in TypesInfo", n.Name.Name))
						  }
					}
				} else if info.EnclosingFuncNode != nil && info.EnclosingFunc == nil {
					 reason := "type info unavailable"
					 if targetPkg != nil && targetPkg.TypesInfo == nil { reason = "TypesInfo is nil" }
					 else if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs == nil { reason = "TypesInfo.Defs is nil" }
					 funcName := "[anonymous]"
					 if info.EnclosingFuncNode.Name != nil { funcName = info.EnclosingFuncNode.Name.Name }
					 addAnalysisError(info, fmt.Errorf("type info for enclosing func '%s' unavailable: %s", funcName, reason))
				}
			case *ast.BlockStmt:
				if info.EnclosingBlock == nil { info.EnclosingBlock = n }
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					if scope := targetPkg.TypesInfo.Scopes[n]; scope != nil {
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					}
				}
			}
		}
	}
	addPackageScope(targetPkg, info)
}


// addPackageScope adds top-level package identifiers to the scope map.
// Includes verification logging.
func addPackageScope(targetPkg *packages.Package, info *AstContextInfo) {
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			pkgScopeNames := pkgScope.Names()
			log.Printf("DEBUG: Adding %d names from package scope '%s': %v", len(pkgScopeNames), targetPkg.Name, pkgScopeNames)
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope) // Use existing helper
		} else {
			addAnalysisError(info, fmt.Errorf("package scope missing for pkg %s", targetPkg.PkgPath))
		}
	} else {
		log.Println("Package type info unavailable for package scope.")
		if targetPkg != nil {
			addAnalysisError(info, fmt.Errorf("package.Types field is nil for pkg %s", targetPkg.PkgPath))
		} else {
			 addAnalysisError(info, errors.New("cannot add package scope: targetPkg is nil"))
		}
	}
}


// addScopeVariables adds identifiers from a types.Scope to the scope map, respecting cursor position for local scopes.
func addScopeVariables(typeScope *types.Scope, cursorPos token.Pos, scopeMap map[string]types.Object) {
	if typeScope == nil { return }
	for _, name := range typeScope.Names() {
		obj := typeScope.Lookup(name)
		if obj == nil { continue }

		include := !cursorPos.IsValid() || !obj.Pos().IsValid() || obj.Pos() < cursorPos

		if include {
			if _, exists := scopeMap[name]; !exists {
				switch obj.(type) {
				case *types.Var, *types.Const, *types.TypeName, *types.Func, *types.Label, *types.PkgName, *types.Builtin, *types.Nil:
					scopeMap[name] = obj
				default:
					log.Printf("Debug: Ignoring object '%s' of type %T in scope.", name, obj)
				}
			}
		}
	}
}


// addSignatureToScope adds parameters and named results from a function signature to the scope map.
func addSignatureToScope(sig *types.Signature, scopeMap map[string]types.Object) {
	if sig == nil { return }
	addTupleToScope(sig.Params(), scopeMap)
	addTupleToScope(sig.Results(), scopeMap) // Add named results to scope
}

// addTupleToScope adds variables from a tuple (params or results) to the scope map.
func addTupleToScope(tuple *types.Tuple, scopeMap map[string]types.Object) {
	if tuple == nil { return }
	for j := 0; j < tuple.Len(); j++ {
		v := tuple.At(j)
		if v != nil && v.Name() != "" {
			if _, exists := scopeMap[v.Name()]; !exists {
				scopeMap[v.Name()] = v
			}
		}
	}
}


// findRelevantComments finds comments immediately preceding the cursor line or associated with the enclosing AST node.
func findRelevantComments(targetFileAST *ast.File, path []ast.Node, cursorPos token.Pos, fset *token.FileSet, info *AstContextInfo) {
	if targetFileAST == nil || fset == nil {
		 addAnalysisError(info, errors.New("cannot find comments: targetFileAST or fset is nil"))
		 return
	}
	cmap := ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, cursorPos, fset)
}

// findCommentsWithMap finds comments immediately preceding cursor or associated with AST path.
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet) []string {
	var comments []string
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		return comments
	}

	cursorLine := fset.Position(cursorPos).Line
	foundPreceding := false
	var precedingComments []string

	// Strategy 1: Find comments ending on the line immediately before the cursor
	for node := range cmap {
		if node == nil { continue }
		for _, cg := range cmap[node] {
			if cg == nil { continue }
			if cg.End().IsValid() && fset.Position(cg.End()).Line == cursorLine-1 {
				for _, c := range cg.List {
					if c != nil { precedingComments = append(precedingComments, c.Text) }
				}
				foundPreceding = true
				break
			}
		}
		if foundPreceding { break }
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

	// Strategy 2: If no preceding comment found, look for Doc comments on enclosing path
	if path != nil {
		for i := 0; i < len(path); i++ {
			node := path[i]
			var docComment *ast.CommentGroup
			switch n := node.(type) {
			case *ast.FuncDecl: docComment = n.Doc
			case *ast.GenDecl: docComment = n.Doc
			case *ast.TypeSpec: docComment = n.Doc
			case *ast.Field: docComment = n.Doc
			case *ast.ValueSpec: docComment = n.Doc
			}
			if docComment != nil {
				for _, c := range docComment.List {
					if c != nil { comments = append(comments, c.Text) }
				}
				goto cleanup
			}
		}
	}

cleanup:
	// Deduplicate final list
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


// buildPreamble constructs the context string for the LLM prompt.
// Uses qualifier directly now. Includes placeholder logic for Step 9 truncation.
func buildPreamble(info *AstContextInfo, qualifier types.Qualifier) string {
	var preamble strings.Builder
	const internalPreambleLimit = 8192 // Generous internal limit

	currentLen := 0
	addToPreamble := func(s string) bool {
		if currentLen+len(s) < internalPreambleLimit { preamble.WriteString(s); currentLen += len(s); return true }
		log.Printf("DEBUG: Preamble internal budget limit reached, skipping part starting with: %.50s", s)
		return false
	}
	addTruncMarker := func(section string) {
		 msg := fmt.Sprintf("//   ... (%s truncated)\n", section)
		 if currentLen+len(msg) < internalPreambleLimit { preamble.WriteString(msg); currentLen += len(msg) }
	}

	if !addToPreamble(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(info.FilePath), info.PackageName)) { return preamble.String() }
	if !formatImportsSection(&preamble, info, addToPreamble, addTruncMarker) { return preamble.String() }
	if !formatEnclosingFuncSection(&preamble, info, qualifier, addToPreamble) { return preamble.String() }
	if !formatCommentsSection(&preamble, info, addToPreamble, addTruncMarker) { return preamble.String() }
	if !formatCursorContextSection(&preamble, info, qualifier, addToPreamble) { return preamble.String() }
	formatScopeSection(&preamble, info, qualifier, addToPreamble, addTruncMarker) // Ignore return, scope is last

	return preamble.String()
}

// formatImportsSection formats the imports part of the preamble.
func formatImportsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string)) bool {
	if len(info.Imports) == 0 { return true }
	header := "// Imports:\n"; if !add(header) { return false }
	count := 0; maxImports := 20
	for _, imp := range info.Imports {
		if imp == nil || imp.Path == nil { continue }
		if count >= maxImports { addTrunc("imports"); return true }
		path := imp.Path.Value; name := ""
		if imp.Name != nil { name = imp.Name.Name + " " }
		line := fmt.Sprintf("//   import %s%s\n", name, path)
		if !add(line) { return false }
		count++
	}
	return true
}

// formatEnclosingFuncSection formats the enclosing function part.
func formatEnclosingFuncSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool) bool {
	funcOrMethod := "Function"; receiverStr := ""
	if info.ReceiverType != "" { funcOrMethod = "Method"; receiverStr = fmt.Sprintf("(%s) ", info.ReceiverType) }
	var funcHeader string
	if info.EnclosingFunc != nil {
		name := info.EnclosingFunc.Name(); sigStr := types.TypeString(info.EnclosingFunc.Type(), qualifier)
		if info.ReceiverType != "" && strings.HasPrefix(sigStr, "func(") { sigStr = "func" + strings.TrimPrefix(sigStr, "func") }
		funcHeader = fmt.Sprintf("// Enclosing %s: %s%s%s\n", funcOrMethod, receiverStr, name, sigStr)
	} else if info.EnclosingFuncNode != nil {
		name := "[anonymous]"; if info.EnclosingFuncNode.Name != nil { name = info.EnclosingFuncNode.Name.Name }
		funcHeader = fmt.Sprintf("// Enclosing %s (AST only): %s%s(...)\n", funcOrMethod, receiverStr, name)
	} else { return true }
	return add(funcHeader)
}

// formatCommentsSection formats relevant comments.
func formatCommentsSection(preamble *strings.Builder, info *AstContextInfo, add func(string) bool, addTrunc func(string)) bool {
	if len(info.CommentsNearCursor) == 0 { return true }
	header := "// Relevant Comments:\n"; if !add(header) { return false }
	count := 0; maxComments := 5
	for _, c := range info.CommentsNearCursor {
		 if count >= maxComments { addTrunc("comments"); return true }
		 cleanComment := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(strings.TrimPrefix(c, "//")), "*/")), "/*"))
		 if len(cleanComment) > 0 {
			 line := fmt.Sprintf("//   %s\n", cleanComment); if !add(line) { return false }; count++
		 }
	}
	return true
}

// formatCursorContextSection formats specific context like calls, selectors, literals. Includes Step 26 change.
func formatCursorContextSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool) bool {
	if info.CallExpr != nil {
		funcName := "[unknown function]"
		switch fun := info.CallExpr.Fun.(type) {
		case *ast.Ident: funcName = fun.Name
		case *ast.SelectorExpr: if fun.Sel != nil { funcName = fun.Sel.Name }
		}
		if !add(fmt.Sprintf("// Inside function call: %s (Arg %d)\n", funcName, info.CallArgIndex+1)) { return false }

		if sig := info.CallExprFuncType; sig != nil {
			if !add(fmt.Sprintf("// Function Signature: %s\n", types.TypeString(sig, qualifier))) { return false }
			params := sig.Params()
			if params != nil && params.Len() > 0 {
				if !add("//   Parameters:\n") { return false }
				for i := 0; i < params.Len(); i++ {
					p := params.At(i); if p == nil { continue }; highlight := ""
					if i == info.CallArgIndex { highlight = " // <-- cursor here" }
					else if sig.Variadic() && i == params.Len()-1 && info.CallArgIndex >= i { highlight = " // <-- cursor here (variadic)" }
					if !add(fmt.Sprintf("//     - %s%s\n", types.ObjectString(p, qualifier), highlight)) { return false }
				}
			} else { if !add("//   Parameters: (none)\n") { return false } }
			results := sig.Results()
			if results != nil && results.Len() > 0 {
				if !add("//   Returns:\n") { return false }
				for i := 0; i < results.Len(); i++ {
					r := results.At(i); if r == nil { continue }
					if !add(fmt.Sprintf("//     - %s\n", types.ObjectString(r, qualifier))) { return false }
				}
			} else { if !add("//   Returns: (none)\n") { return false } }
		} else {
			if !add("// Function Signature: (unknown - type analysis failed for call expression)\n") { return false }
		}
		return true
	}

	if info.SelectorExpr != nil {
		selName := ""; if info.SelectorExpr.Sel != nil { selName = info.SelectorExpr.Sel.Name }
		typeName := "(unknown - type analysis failed for base expression)"
		if info.SelectorExprType != nil { typeName = types.TypeString(info.SelectorExprType, qualifier) }
		if !add(fmt.Sprintf("// Selector context: expr type = %s (selecting '%s')\n", typeName, selName)) { return false }

		if info.SelectorExprType != nil {
			members := listTypeMembers(info.SelectorExprType, info.SelectorExpr.X, qualifier) // Use refactored listTypeMembers
			var fields, methods []MemberInfo
			if members != nil {
				for _, member := range members {
					if member.Kind == FieldMember { fields = append(fields, member) }
					if member.Kind == MethodMember { methods = append(methods, member) }
				}
			} else { log.Printf("DEBUG: listTypeMembers returned nil for type %s", typeName) }

			if len(fields) > 0 {
				if !add("//   Available Fields:\n") { return false }
				sort.Slice(fields, func(i, j int) bool { return fields[i].Name < fields[j].Name })
				for _, field := range fields {
					if !add(fmt.Sprintf("//     - %s %s\n", field.Name, field.TypeString)) { return false }
				}
			}
			if len(methods) > 0 {
				if !add("//   Available Methods:\n") { return false }
				sort.Slice(methods, func(i, j int) bool { return methods[i].Name < methods[j].Name })
				for _, method := range methods {
					methodSig := strings.TrimPrefix(method.TypeString, "func")
					if !add(fmt.Sprintf("//     - %s%s\n", method.Name, methodSig)) { return false }
				}
			}
			if len(fields) == 0 && len(methods) == 0 {
				 if members == nil {
					  if !add("//   (Could not determine members)\n") { return false}
				  } else {
					  if !add("//   (No exported fields or methods found)\n") { return false }
				  }
			}
		} else {
			if !add("//   (Cannot list members: type analysis failed for base expression)\n") { return false }
		}
		return true
	}

	if info.CompositeLit != nil {
		typeName := "(unknown - type analysis failed for literal)"
		if info.CompositeLitType != nil { typeName = types.TypeString(info.CompositeLitType, qualifier) }
		if !add(fmt.Sprintf("// Inside composite literal of type: %s\n", typeName)) { return false }

		if info.CompositeLitType != nil {
			var st *types.Struct; currentType := info.CompositeLitType.Underlying()
			if ptr, ok := currentType.(*types.Pointer); ok { if ptr.Elem() != nil { currentType = ptr.Elem().Underlying()} else { currentType = nil } }
			st, _ = currentType.(*types.Struct)

			if st != nil {
				presentFields := make(map[string]bool)
				for _, elt := range info.CompositeLit.Elts { if kv, ok := elt.(*ast.KeyValueExpr); ok { if kid, ok := kv.Key.(*ast.Ident); ok { presentFields[kid.Name] = true } } }
				var missingFields []string
				for i := 0; i < st.NumFields(); i++ { field := st.Field(i); if field != nil && field.Exported() && !presentFields[field.Name()] { missingFields = append(missingFields, types.ObjectString(field, qualifier)) } }
				if len(missingFields) > 0 {
					if !add("//   Missing Exported Fields (candidates for completion):\n") { return false }
					sort.Strings(missingFields)
					for _, fieldStr := range missingFields { if !add(fmt.Sprintf("//     - %s\n", fieldStr)) { return false } }
				} else { if !add("//   (All exported fields may be present or none missing)\n") { return false } }
			} else { if !add("//   (Underlying type is not a struct)\n") { return false } }
		} else {
			if !add("//   (Cannot determine missing fields: type analysis failed)\n") { return false }
		}
		return true
	}

	if info.IdentifierAtCursor != nil {
		identName := info.IdentifierAtCursor.Name
		if info.IdentifierType != nil {
			if !add(fmt.Sprintf("// Identifier at cursor: %s (Type: %s)\n", identName, types.TypeString(info.IdentifierType, qualifier))) { return false }
		} else {
			if !add(fmt.Sprintf("// Identifier at cursor: %s (Type unknown)\n", identName)) { return false }
		}
		return true
	}

	return true
}


// formatScopeSection formats variables/etc in scope.
func formatScopeSection(preamble *strings.Builder, info *AstContextInfo, qualifier types.Qualifier, add func(string) bool, addTrunc func(string)) bool {
	if len(info.VariablesInScope) == 0 { return true }
	header := "// Variables/Constants/Types in Scope:\n"; if !add(header) { return false }
	var items []string
	for name := range info.VariablesInScope { obj := info.VariablesInScope[name]; items = append(items, fmt.Sprintf("//   %s\n", types.ObjectString(obj, qualifier))) }
	sort.Strings(items)
	count := 0; maxScopeItems := 30
	for _, item := range items {
		if count >= maxScopeItems { addTrunc("scope"); return true }
		if !add(item) { return false }; count++
	}
	return true
}


// calculateCursorPos implementation retained and refined
func calculateCursorPos(file *token.File, line, col int) (token.Pos, error) {
	if line <= 0 { return token.NoPos, fmt.Errorf("invalid line number: %d (must be >= 1)", line) }
	if col <= 0 { return token.NoPos, fmt.Errorf("invalid column number: %d (must be >= 1)", col) }
	if file == nil { return token.NoPos, errors.New("invalid token.File (nil)") }

	fileLineCount := file.LineCount()
	if line > fileLineCount {
		if line == fileLineCount+1 && col == 1 {
			 log.Printf("DEBUG: Cursor on newline after last line (%d), using EOF offset %d", fileLineCount, file.Size())
			 return file.Pos(file.Size()), nil
		 }
		return token.NoPos, fmt.Errorf("line number %d exceeds file line count %d", line, fileLineCount)
	}
	lineStartPos := file.LineStart(line)
	if !lineStartPos.IsValid() { return token.NoPos, fmt.Errorf("cannot get start offset for line %d in file '%s'", line, file.Name()) }

	lineStartOffset := file.Offset(lineStartPos)
	cursorOffset := lineStartOffset + col - 1

	lineEndOffset := file.Size()
	if line < fileLineCount {
		 nextLineStartPos := file.LineStart(line + 1)
		 if nextLineStartPos.IsValid() { lineEndOffset = file.Offset(nextLineStartPos) }
	}

	finalOffset := cursorOffset
	if cursorOffset < lineStartOffset {
		 log.Printf("Warning: column %d resulted in offset %d before line start %d. Clamping to line start.", col, cursorOffset, lineStartOffset)
		 finalOffset = lineStartOffset
	} else if cursorOffset > lineEndOffset {
		log.Printf("Warning: column %d resulted in offset %d beyond line end %d. Clamping to line end.", col, cursorOffset, lineEndOffset)
		finalOffset = lineEndOffset
	}

	pos := file.Pos(finalOffset)
	if !pos.IsValid() {
		log.Printf("Error: Clamped offset %d resulted in invalid token.Pos. Using line start %d.", finalOffset, lineStartPos)
		return lineStartPos, fmt.Errorf("failed to calculate valid token.Pos for offset %d", finalOffset)
	}
	return pos, nil
}


// findContextNodes includes fix for posStr helper.
func findContextNodes(path []ast.Node, cursorPos token.Pos, pkg *packages.Package, fset *token.FileSet, info *AstContextInfo) {
	if len(path) == 0 || fset == nil {
		log.Println("Cannot find context nodes: Missing AST path or FileSet.")
		if fset == nil { addAnalysisError(info, errors.New("Fset is nil in findContextNodes")) }
		return
	}
	posStr := func(p token.Pos) string {
		if p.IsValid() { return fset.Position(p).String() }
		return fmt.Sprintf("Pos(%d)", p)
	}

	hasTypeInfo := pkg != nil && pkg.TypesInfo != nil
	var typesMap map[ast.Expr]types.TypeAndValue
	var defsMap map[*ast.Ident]types.Object
	var usesMap map[*ast.Ident]types.Object
	if hasTypeInfo {
		 typesMap = pkg.TypesInfo.Types; defsMap = pkg.TypesInfo.Defs; usesMap = pkg.TypesInfo.Uses
		 if typesMap == nil { addAnalysisError(info, errors.New("type info map 'Types' is nil")) }
		 if defsMap == nil { addAnalysisError(info, errors.New("type info map 'Defs' is nil")) }
		 if usesMap == nil { addAnalysisError(info, errors.New("type info map 'Uses' is nil")) }
	} else if pkg == nil { addAnalysisError(info, errors.New("cannot perform type analysis: target package is nil"))
	} else { addAnalysisError(info, errors.New("cannot perform type analysis: pkg.TypesInfo is nil")) }

	innermostNode := path[0]

	// CompositeLit Check
	if compLit, ok := innermostNode.(*ast.CompositeLit); ok && cursorPos >= compLit.Lbrace && cursorPos <= compLit.Rbrace {
		info.CompositeLit = compLit; log.Printf("Cursor is inside CompositeLit at pos %s", posStr(compLit.Pos()))
		if hasTypeInfo && typesMap != nil { if tv, ok := typesMap[compLit]; ok { info.CompositeLitType = tv.Type; if info.CompositeLitType == nil { addAnalysisError(info, fmt.Errorf("composite literal type resolved to nil at %s", posStr(compLit.Pos()))) } } else { addAnalysisError(info, fmt.Errorf("missing type info for composite literal at %s", posStr(compLit.Pos()))) } } else if hasTypeInfo && typesMap == nil { addAnalysisError(info, errors.New("cannot analyze composite literal: Types map is nil"))}
		return
	}
	// CallExpr Check
	callExpr, callExprOk := path[0].(*ast.CallExpr); if !callExprOk && len(path) > 1 { callExpr, callExprOk = path[1].(*ast.CallExpr) }
	if callExprOk && cursorPos > callExpr.Lparen && cursorPos <= callExpr.Rparen {
		info.CallExpr = callExpr; info.CallArgIndex = calculateArgIndex(callExpr.Args, cursorPos)
		log.Printf("Cursor likely at argument index %d in call at %s", info.CallArgIndex, posStr(callExpr.Pos()))
		if hasTypeInfo && typesMap != nil { if tv, ok := typesMap[callExpr.Fun]; ok && tv.Type != nil { if sig, ok := tv.Type.Underlying().(*types.Signature); ok { info.CallExprFuncType = sig; info.ExpectedArgType = determineExpectedArgType(sig, info.CallArgIndex) } else { addAnalysisError(info, fmt.Errorf("type of call func (%T) at %s is %T, not signature", callExpr.Fun, posStr(callExpr.Fun.Pos()), tv.Type)) } } else { if ok && tv.Type == nil { addAnalysisError(info, fmt.Errorf("type info resolved to nil for call func (%T) at %s", callExpr.Fun, posStr(callExpr.Fun.Pos()))) } else { addAnalysisError(info, fmt.Errorf("missing type info for call func (%T) at %s", callExpr.Fun, posStr(callExpr.Fun.Pos()))) } } } else if hasTypeInfo && typesMap == nil { addAnalysisError(info, errors.New("cannot analyze call expr: Types map is nil"))}
		return
	}
	// SelectorExpr Check
	for i := 0; i < len(path) && i < 2; i++ {
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok && cursorPos > selExpr.X.End() {
			info.SelectorExpr = selExpr
			if hasTypeInfo && typesMap != nil { if tv, ok := typesMap[selExpr.X]; ok { info.SelectorExprType = tv.Type; if tv.Type == nil { addAnalysisError(info, fmt.Errorf("missing type for selector base expr (%T) starting at %s", selExpr.X, posStr(selExpr.X.Pos()))) } } else { addAnalysisError(info, fmt.Errorf("missing type info entry for selector base expr (%T) starting at %s", selExpr.X, posStr(selExpr.X.Pos()))) } } else if hasTypeInfo && typesMap == nil { addAnalysisError(info, errors.New("cannot analyze selector expr: Types map is nil"))}
			return
		}
	}
	// Identifier Check (as fallback)
	var ident *ast.Ident
	 if id, ok := path[0].(*ast.Ident); ok && cursorPos == id.End() { ident = id } else if len(path) > 1 { if id, ok := path[1].(*ast.Ident); ok && cursorPos >= id.Pos() && cursorPos <= id.End() { if _, pIsSel := path[0].(*ast.SelectorExpr); !pIsSel || path[0].(*ast.SelectorExpr).Sel != id { ident = id } } }
	if ident != nil {
		info.IdentifierAtCursor = ident
		if hasTypeInfo {
			var obj types.Object
			if usesMap != nil { obj = usesMap[ident] }
			if obj == nil && defsMap != nil { obj = defsMap[ident] }
			if obj != nil { info.IdentifierObject = obj; info.IdentifierType = obj.Type(); if info.IdentifierType == nil { addAnalysisError(info, fmt.Errorf("object '%s' at %s found but type is nil", obj.Name(), posStr(obj.Pos()))) }
			} else { if typesMap != nil { if tv, ok := typesMap[ident]; ok && tv.Type != nil { info.IdentifierType = tv.Type } else { if defsMap != nil && usesMap != nil { addAnalysisError(info, fmt.Errorf("missing object and type info for identifier '%s' at %s", ident.Name, posStr(ident.Pos()))) } } } else if defsMap == nil && usesMap == nil { addAnalysisError(info, fmt.Errorf("missing object info for identifier '%s' at %s (defs/uses/types maps nil)", ident.Name, posStr(ident.Pos()))) } else { addAnalysisError(info, fmt.Errorf("missing object info for identifier '%s' at %s", ident.Name, posStr(ident.Pos()))) } }
		} else { addAnalysisError(info, errors.New("missing type info for identifier analysis")) }
	} else { log.Println("No specific identifier found at cursor position.") }
}



// calculateArgIndex determines which argument the cursor is likely positioned at.
func calculateArgIndex(args []ast.Expr, cursorPos token.Pos) int {
	if len(args) == 0 { return 0 }
	for i, arg := range args {
		 if arg == nil { continue }
		 argStart := arg.Pos(); argEnd := arg.End()
		 slotStart := argStart
		 if i > 0 && args[i-1] != nil { slotStart = args[i-1].End() + 1 }
		 if cursorPos >= slotStart && cursorPos <= argEnd { return i }
		 if cursorPos > arg.End()+1 { if i == len(args)-1 { return i + 1 } } else if cursorPos > arg.End() { return i + 1 }
	}
	if len(args) > 0 && args[0] != nil && cursorPos < args[0].Pos() { return 0 }
	log.Printf("DEBUG: calculateArgIndex fallback, returning 0.")
	return 0
}

// determineExpectedArgType finds the type expected for a given argument index in a signature.
func determineExpectedArgType(sig *types.Signature, argIndex int) types.Type {
	if sig == nil || argIndex < 0 { return nil }
	params := sig.Params(); if params == nil { return nil }
	numParams := params.Len(); if numParams == 0 { return nil }
	if sig.Variadic() {
		if argIndex >= numParams-1 {
			lastParam := params.At(numParams - 1); if lastParam == nil { return nil }
			if slice, ok := lastParam.Type().(*types.Slice); ok { return slice.Elem() }
			log.Printf("Warning: Variadic parameter %s is not a slice type (%T)", lastParam.Name(), lastParam.Type()); return nil
		} else if argIndex < numParams-1 {
			 param := params.At(argIndex); if param == nil { return nil }; return param.Type()
		}
	} else { if argIndex < numParams { param := params.At(argIndex); if param == nil { return nil }; return param.Type() } }
	return nil
}


// MemberKind defines the type of member (field or method).
type MemberKind string
const ( FieldMember MemberKind = "field"; MethodMember MemberKind = "method"; OtherMember MemberKind = "other" )
// MemberInfo holds structured information about a type member.
type MemberInfo struct { Name string; Kind MemberKind; TypeString string; }

// listTypeMembers returns structured info about fields/methods of a type. (Refactored Step 26)
func listTypeMembers(typ types.Type, expr ast.Expr, qualifier types.Qualifier) []MemberInfo {
	if typ == nil { return nil }
	var members []MemberInfo // Return slice of MemberInfo directly

	currentType := typ
	isPointer := false
	if ptr, ok := typ.(*types.Pointer); ok {
		if ptr.Elem() == nil { return nil }
		currentType = ptr.Elem()
		isPointer = true
	}
	if currentType == nil { return nil }
	underlying := currentType.Underlying()
	if underlying == nil { return nil }

	// Basic/Map/Slice/Chan placeholders
	if basic, ok := underlying.(*types.Basic); ok {
		members = append(members, MemberInfo{Name: basic.String(), Kind: OtherMember, TypeString: "(basic type)"})
	} else if _, ok := underlying.(*types.Map); ok {
		members = append(members, MemberInfo{Name:"// map operations", Kind: OtherMember, TypeString:"make, len, delete, range"})
	} else if _, ok := underlying.(*types.Slice); ok {
		members = append(members, MemberInfo{Name:"// slice operations", Kind: OtherMember, TypeString:"make, len, cap, append, copy, range"})
	} else if _, ok := underlying.(*types.Chan); ok {
		members = append(members, MemberInfo{Name:"// channel operations", Kind: OtherMember, TypeString:"make, len, cap, close, send (<-), receive (<-)"})
	}

	// Struct fields - Use helper
	if st, ok := underlying.(*types.Struct); ok {
		members = append(members, listStructFields(st, qualifier)...)
	}

	// Interface methods
	if iface, ok := underlying.(*types.Interface); ok {
		for i := 0; i < iface.NumExplicitMethods(); i++ {
			method := iface.ExplicitMethod(i)
			if method != nil && method.Exported() {
				members = append(members, MemberInfo{
					Name: method.Name(),
					Kind: MethodMember,
					TypeString: types.TypeString(method.Type(), qualifier),
				})
			}
		}
		for i := 0; i < iface.NumEmbeddeds(); i++ {
			embeddedType := iface.EmbeddedType(i)
			if embeddedType != nil {
				members = append(members, MemberInfo{
					Name: "// embeds",
					Kind: OtherMember,
					TypeString: types.TypeString(embeddedType, qualifier),
				})
			}
		}
	}

	// Method sets (Value receiver methods for original type)
	mset := types.NewMethodSet(typ) // Use original type `typ`
	for i := 0; i < mset.Len(); i++ {
		sel := mset.At(i)
		if sel != nil {
			methodObj := sel.Obj()
			if method, ok := methodObj.(*types.Func); ok {
				if method != nil && method.Exported() {
					members = append(members, MemberInfo{
						Name: method.Name(),
						Kind: MethodMember,
						TypeString: types.TypeString(method.Type(), qualifier),
					})
				}
			} else if methodObj != nil { log.Printf("Warning: Object in value method set is not *types.Func: %T", methodObj) }
		}
	}

	// Add pointer receiver methods if base type is not already a pointer
	if !isPointer {
		// Use currentType which is already dereferenced if original was pointer
		if named, ok := currentType.(*types.Named); ok { // Check if it's a named type
			if named != nil {
				ptrToNamed := types.NewPointer(named) // Pointer to the base named type
				msetPtr := types.NewMethodSet(ptrToNamed)
				// Keep track of methods already added from value receiver
				existingMethods := make(map[string]struct{})
				for _, m := range members {
					if m.Kind == MethodMember {
						 existingMethods[m.Name] = struct{}{}
					}
				}

				for i := 0; i < msetPtr.Len(); i++ {
					sel := msetPtr.At(i)
					if sel != nil {
						methodObj := sel.Obj()
						if method, ok := methodObj.(*types.Func); ok {
							if method != nil && method.Exported() {
								if _, exists := existingMethods[method.Name()]; !exists { // Add only if not already present
									members = append(members, MemberInfo{
										Name: method.Name(),
										Kind: MethodMember,
										TypeString: types.TypeString(method.Type(), qualifier),
									})
								}
							}
						} else if methodObj != nil {
							log.Printf("Warning: Object in pointer method set is not *types.Func: %T", methodObj)
						}
					}
				} // End for msetPtr
			} // End if named != nil
		} // End if named, ok
	} // End if !isPointer


	// Deduplicate final list based on Kind and Name
	if len(members) > 0 {
		seen := make(map[string]struct{})
		uniqueMembers := make([]MemberInfo, 0, len(members))
		for _, m := range members {
			key := string(m.Kind) + ":" + m.Name // Use kind and name for uniqueness
			if _, ok := seen[key]; !ok {
				seen[key] = struct{}{}
				uniqueMembers = append(uniqueMembers, m)
			}
		}
		members = uniqueMembers
	}
	return members
}

// listStructFields extracts exported fields from a struct type. (Helper for Step 26)
func listStructFields(st *types.Struct, qualifier types.Qualifier) []MemberInfo {
	var fields []MemberInfo
	if st == nil { return nil }
	for i := 0; i < st.NumFields(); i++ {
		field := st.Field(i)
		if field != nil && field.Exported() {
			fields = append(fields, MemberInfo{
				Name:       field.Name(),
				Kind:       FieldMember,
				TypeString: types.TypeString(field.Type(), qualifier),
			})
		}
	}
	return fields
}


// logAnalysisErrors logs collected non-fatal errors.
func logAnalysisErrors(errs []error) {
	if len(errs) > 0 {
		// Use errors.Join for potentially better formatting if available
		combinedErr := errors.Join(errs...)
		log.Printf("Context analysis completed with %d non-fatal error(s): %v", len(errs), combinedErr)
	}
}
// addAnalysisError adds a non-fatal error to the info struct, avoiding simple duplicates.
func addAnalysisError(info *AstContextInfo, err error) {
	if err != nil && info != nil {
		// Avoid adding duplicate error messages
		 errMsg := err.Error()
		 for _, existing := range info.AnalysisErrors {
			  if existing.Error() == errMsg { return }
		  }
		log.Printf("Analysis Warning: %v", err) // Log immediately
		info.AnalysisErrors = append(info.AnalysisErrors, err)
	}
}


// =============================================================================
// Spinner & File Helpers (Spinner Exported, others internal)
// =============================================================================
// Spinner implementation retained
type Spinner struct {
	chars    []string
	message  string // Current message to display
	index    int
	mu       sync.Mutex // Protects message and index
	stopChan chan struct{}
	doneChan chan struct{} // Used for graceful shutdown confirmation
	running  bool
}
func NewSpinner() *Spinner {
	return &Spinner{
		chars: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
		index: 0,
	}
}
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
			s.mu.Lock()
			isRunning := s.running
			s.running = false // Mark as not running *before* potentially clearing line
			s.mu.Unlock()
			if isRunning {
				fmt.Fprintf(os.Stderr, "\r\033[K") // Clear line on stop only if it was running
			}
			select {
			case s.doneChan <- struct{}{}: // Signal done
			default: // Avoid blocking if Stop already timed out
			}
			close(s.doneChan) // Close after signaling
		}()

		for {
			select {
			case <-s.stopChan:
				return // Exit goroutine when stopChan is closed
			case <-ticker.C:
				s.mu.Lock()
				if !s.running { // Check running flag inside loop as well
					s.mu.Unlock()
					return // Exit if stopped between ticks
				}
				char := s.chars[s.index]
				msg := s.message
				s.index = (s.index + 1) % len(s.chars)
				// Use stderr for spinner to avoid interfering with completion output on stdout
				fmt.Fprintf(os.Stderr, "\r\033[K%s%s%s %s", ColorCyan, char, ColorReset, msg)
				s.mu.Unlock()
			}
		}
	}()
}
func (s *Spinner) UpdateMessage(newMessage string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running { // Only update if actually running
		s.message = newMessage
	}
}
func (s *Spinner) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return // Already stopped
	}
	select {
	case <-s.stopChan: // Already closed? Should not happen if running=true
	default:
		close(s.stopChan) // Close channel to signal goroutine
	}
	doneChan := s.doneChan // Read under lock
	s.mu.Unlock()          // Unlock before potentially waiting

	if doneChan != nil {
		// Wait with timeout for the goroutine to finish cleanup
		select {
		case <-doneChan: // Wait for goroutine to signal done
		case <-time.After(500 * time.Millisecond):
			log.Println("Warning: Timeout waiting for spinner goroutine cleanup")
		}
	}
	// Ensure line is cleared finally, regardless of goroutine state after timeout
	fmt.Fprintf(os.Stderr, "\r\033[K")
}


// extractSnippetContext implementation retained
func extractSnippetContext(filename string, row, col int) (SnippetContext, error) {
	var ctx SnippetContext
	contentBytes, err := os.ReadFile(filename)
	if err != nil {
		return ctx, fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	content := string(contentBytes)
	fset := token.NewFileSet()
	// Use filename directly, Abs path handled in Analyze now
	// Use Base=1 for consistency with FileSet usage elsewhere if needed
	file := fset.AddFile(filename, 1, len(contentBytes))
	if file == nil {
		return ctx, fmt.Errorf("failed to add file '%s' to fileset", filename)
	}

	// Use refined cursor position calculator
	cursorPos, posErr := calculateCursorPos(file, row, col)
	if posErr != nil {
		return ctx, fmt.Errorf("cannot determine valid cursor position: %w", posErr)
	}
	if !cursorPos.IsValid() {
		// calculateCursorPos should return error if invalid, but double check
		return ctx, fmt.Errorf("invalid cursor position calculated (Pos: %d)", cursorPos)
	}

	offset := file.Offset(cursorPos)
	// Clamp offset for safety, though calculateCursorPos should already do this
	if offset < 0 { offset = 0 }
	if offset > len(content) { offset = len(content) }

	ctx.Prefix = content[:offset]
	ctx.Suffix = content[offset:]

	// Extract full line (improved bounds checking)
	lineStartPos := file.LineStart(row)
	if !lineStartPos.IsValid() {
		log.Printf("Warning: Could not get start position for line %d in %s", row, filename)
		// Return context without FullLine, but not as an error
		return ctx, nil
	}
	startOffset := file.Offset(lineStartPos)

	lineEndOffset := file.Size() // Default for last line
	fileLineCount := file.LineCount()
	if row < fileLineCount {
		nextLineStartPos := file.LineStart(row + 1)
		if nextLineStartPos.IsValid() {
			lineEndOffset = file.Offset(nextLineStartPos)
		}
	}

	// Check calculated offsets are valid relative to content length
	if startOffset >= 0 && lineEndOffset >= startOffset && lineEndOffset <= len(content) {
		// Extract line, trim trailing newline characters (\n or \r\n)
		lineContent := content[startOffset:lineEndOffset]
		lineContent = strings.TrimRight(lineContent, "\n")
		lineContent = strings.TrimRight(lineContent, "\r")
		ctx.FullLine = lineContent
	} else {
		log.Printf("Warning: Could not extract full line for row %d (start %d, end %d, content len %d) in %s",
			row, startOffset, lineEndOffset, len(content), filename)
	}
	return ctx, nil
}


// --- Cache Helper Functions (Added Step 19/23, Modified Step 25) ---
// calculateGoModHash implementation retained
func calculateGoModHash(dir string) string {
	goModPath := filepath.Join(dir, "go.mod")
	f, err := os.Open(goModPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) { return "no-gomod" }
		log.Printf("Warning: Error reading %s: %v", goModPath, err)
		return "read-error"
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		log.Printf("Warning: Error hashing %s: %v", goModPath, err)
		return "hash-error"
	}
	return hex.EncodeToString(h.Sum(nil))
}

// calculateInputHashes implementation retained
func calculateInputHashes(dir string, pkg *packages.Package) (map[string]string, error) {
	hashes := make(map[string]string)
	filesToHash := make(map[string]struct{}) // Use map to prevent duplicates

	// Add go.mod and go.sum first if they exist
	goModPath := filepath.Join(dir, "go.mod")
	if _, err := os.Stat(goModPath); err == nil {
		absPath, absErr := filepath.Abs(goModPath)
		if absErr == nil { filesToHash[absPath] = struct{}{} } else { log.Printf("Warning: Could not get absolute path for %s: %v", goModPath, absErr) }
	} else if !errors.Is(err, os.ErrNotExist) {
		 return nil, fmt.Errorf("failed to stat %s: %w", goModPath, err) // Error if stat fails unexpectedly
	}
	goSumPath := filepath.Join(dir, "go.sum")
	if _, err := os.Stat(goSumPath); err == nil {
		absPath, absErr := filepath.Abs(goSumPath)
		if absErr == nil { filesToHash[absPath] = struct{}{} } else { log.Printf("Warning: Could not get absolute path for %s: %v", goSumPath, absErr) }
	} else if !errors.Is(err, os.ErrNotExist) {
		 return nil, fmt.Errorf("failed to stat %s: %w", goSumPath, err)
	}

	// Add files from package load results if available
	filesFromPkg := false
	if pkg != nil && len(pkg.CompiledGoFiles) > 0 {
		 filesFromPkg = true
		 for _, fpath := range pkg.CompiledGoFiles {
			  absPath, err := filepath.Abs(fpath) // Ensure path is absolute
			  if err == nil {
				  filesToHash[absPath] = struct{}{}
			  } else {
				  log.Printf("Warning: Could not get absolute path for %s: %v", fpath, err)
			  }
		  }
	}

	// If package data wasn't available (e.g., validating cache, pkg=nil), scan dir
	if !filesFromPkg {
		log.Printf("DEBUG: Calculating input hashes by scanning directory %s (pkg info unavailable)", dir)
		entries, err := os.ReadDir(dir)
		if err != nil { return nil, fmt.Errorf("failed to scan directory %s for hashing: %w", dir, err) }
		for _, entry := range entries {
			// Include only .go files for dir scan fallback
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") {
				 absPath := filepath.Join(dir, entry.Name())
				 absPath, absErr := filepath.Abs(absPath) // Ensure absolute
				 if absErr == nil { filesToHash[absPath] = struct{}{} } else { log.Printf("Warning: Could not get absolute path for %s: %v", entry.Name(), absErr)}
			}
		}
	}


	// Calculate hashes for all collected files
	for absPath := range filesToHash {
		// Use relative path from dir as the key in the map for consistency
		relPath, err := filepath.Rel(dir, absPath)
		if err != nil {
			log.Printf("Warning: Could not get relative path for %s in %s: %v", absPath, dir, err)
			relPath = filepath.Base(absPath) // Use base name as fallback key
		}
		hash, err := hashFileContent(absPath)
		if err != nil {
			// If a file cannot be hashed (e.g., permission error), fail the whole process
			return nil, fmt.Errorf("failed to hash input file %s: %w", absPath, err)
		}
		hashes[relPath] = hash
	}
	return hashes, nil
}

// hashFileContent implementation retained
func hashFileContent(filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil { return "", err }
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil { return "", err }
	return hex.EncodeToString(h.Sum(nil)), nil
}

// compareFileHashes implementation retained
func compareFileHashes(current, cached map[string]string) bool {
	if len(current) != len(cached) {
		log.Printf("DEBUG: Cache invalid: File count mismatch (%d vs %d cached)", len(current), len(cached))
		return false
	}
	for relPath, currentHash := range current {
		cachedHash, ok := cached[relPath]
		if !ok {
			log.Printf("DEBUG: Cache invalid: File %s missing in cache.", relPath)
			return false
		}
		if currentHash != cachedHash {
			log.Printf("DEBUG: Cache invalid: Hash mismatch for file %s.", relPath)
			return false
		}
	}
	for relPath := range cached {
		 if _, ok := current[relPath]; !ok {
			 log.Printf("DEBUG: Cache invalid: Cached file %s not found in current inputs.", relPath)
			 return false
		 }
	 }
	return true
}

// encodeTypesPackage uses exportdata.Write (Step 25 Implementation)
func encodeTypesPackage(fset *token.FileSet, pkg *types.Package) ([]byte, error) {
	if fset == nil || pkg == nil {
		return nil, errors.New("encodeTypesPackage requires non-nil fset and pkg")
	}
	var buf bytes.Buffer
	// Use exportdata.Write - it handles the complexity internally.
	err := exportdata.Write(&buf, fset, pkg)
	if err != nil {
		return nil, fmt.Errorf("exportdata.Write failed: %w", err)
	}
	log.Printf("DEBUG: Serialized types.Package using exportdata (size %d)", buf.Len())
	return buf.Bytes(), nil
}

// decodeTypesPackage uses exportdata.Read (Step 25 Implementation)
func decodeTypesPackage(data []byte, fset *token.FileSet, pkgPathHint string) (*types.Package, error) {
	if fset == nil {
		return nil, errors.New("decodeTypesPackage requires non-nil fset")
	}
	if len(data) == 0 {
		return nil, errors.New("cannot decode empty types data")
	}
	// Provide a basic importer that fails for now if dependencies are needed.
	importsMap := make(map[string]*types.Package)
	importerFunc := func(imports map[string]*types.Package, path string) (*types.Package, error) {
		// Required signature for exportdata.Read importer
		if pkg, ok := imports[path]; ok { return pkg, nil }
		if path == "unsafe" { return types.Unsafe, nil}
		return nil, fmt.Errorf("dependency resolution (%s) not supported during cache decoding", path)
	}

	// Read the package data. Use pkgPathHint.
	// Note: The importerFunc signature used by exportdata.Read is simpler than types.ImporterFrom
	pkg, err := exportdata.Read(bytes.NewReader(data), fset, importsMap, pkgPathHint, importerFunc)
	if err != nil {
		return nil, fmt.Errorf("exportdata.Read failed: %w", err)
	}
	// Note: The resulting pkg.TypesInfo will likely be empty or incomplete after exportdata.Read.
	log.Printf("DEBUG: Decoded types.Package using exportdata for path %s. NOTE: TypesInfo needs reconstruction.", pkg.Path())
	return pkg, nil
}


// parsePackageFiles implementation retained
func parsePackageFiles(ctx context.Context, dir string, fset *token.FileSet) ([]*ast.File, []error) {
	pkgs, err := packages.Load(&packages.Config{
		Context: ctx, Mode:    packages.NeedFiles | packages.NeedSyntax,
		Dir:     dir, Fset:    fset,
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			const mode = parser.ParseComments | parser.AllErrors
			file, err := parser.ParseFile(fset, filename, src, mode)
			if err != nil { log.Printf("Parser error in %s during re-parse: %v", filename, err) }
			return file, nil
		},
		Tests: true,
	}, ".")
	if err != nil { return nil, []error{fmt.Errorf("re-parsing package failed: %w", err)} }
	var files []*ast.File; var errs []error
	if len(pkgs) > 0 {
		files = pkgs[0].Syntax
		for _, e := range pkgs[0].Errors { errs = append(errs, e) }
	} else { errs = append(errs, fmt.Errorf("no packages found during re-parse in %s", dir)) }
	return files, errs
}

// keysFromAstFileMap implementation retained
func keysFromAstFileMap(files []*ast.File, fset *token.FileSet) []string {
	if fset == nil { return nil }
	keys := make([]string, 0, len(files)); processed := make(map[string]struct{})
	for _, astFile := range files {
		if astFile != nil {
			 tokenFile := fset.File(astFile.Pos())
			 if tokenFile != nil && tokenFile.Name() != "" {
				  fname := tokenFile.Name()
				  if _, seen := processed[fname]; !seen { keys = append(keys, fname); processed[fname] = struct{}{} }
			 }
		 }
	}
	return keys
}

// deleteCacheEntry implementation retained
func (a *GoPackagesAnalyzer) deleteCacheEntry(cacheKey []byte) {
	if a.db == nil { return }
	err := a.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(cacheBucketName)
		if b == nil { return nil }
		if b.Get(cacheKey) == nil { return nil } // Key doesn't exist
		log.Printf("DEBUG: Deleting cache entry for key: %s", string(cacheKey))
		return b.Delete(cacheKey)
	})
	if err != nil { log.Printf("Warning: Failed to delete cache entry %s: %v", string(cacheKey), err) }
}
