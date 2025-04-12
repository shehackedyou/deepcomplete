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

	// "go/scanner" // No longer needed
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
// Constants & Types
// =============================================================================

const (
	defaultOllamaURL = "http://localhost:11434"
	defaultModel     = "deepseek-coder-r2"
	// Updated prompt template to better guide the LLM with context
	promptTemplate = `<s>[INST] <<SYS>>
You are an AI coding assistant specializing in Go.
Complete the Go code based on the provided context (enclosing function, variables in scope, relevant comments, identifier/call/selector under cursor).
Adhere strictly to Go syntax and common practices.
Output ONLY the code completion, without any introductory text, comments, or explanations.
Match the indentation of the surrounding code.
<</SYS>>

CONTEXT:
%s

CODE SNIPPET TO COMPLETE:
` + "```go\n%s\n```" + `
[/INST]`
	maxRetries       = 3 // Retries only used for streaming completion now
	retryDelay       = 2 * time.Second
	maxContentLength = 2048 // Max length for the CODE SNIPPET part
	defaultMaxTokens = 256
	// DefaultStop is exported for potential use in CLI default flags
	DefaultStop        = "\n" // Common stop sequences for code completion
	defaultTemperature = 0.1
)

// Config holds the configuration for the autocompletion. Exported.
type Config struct {
	OllamaURL      string
	Model          string
	PromptTemplate string
	MaxTokens      int
	Stop           []string
	Temperature    float64
	Rules          []string // Custom rules (currently unused in core logic)
	// UseLexer flag removed
	UseAst bool // Use AST and Type analysis for context (preferred)
}

// OllamaResponse represents the streaming response structure (internal).
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"` // Check this field in the stream
}

// Middleware defines a function type for processing input or output. Exported.
type Middleware func(input string) (string, error)

// LexerResult removed as lexer path is removed.

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

// =============================================================================
// Variables
// =============================================================================

var (
	// DefaultConfig enables AST context by default. Exported.
	DefaultConfig = Config{
		OllamaURL:      defaultOllamaURL,
		Model:          defaultModel,
		PromptTemplate: promptTemplate,
		MaxTokens:      defaultMaxTokens,
		Stop:           []string{DefaultStop, "}", "//", "/*"}, // Use exported DefaultStop
		Temperature:    defaultTemperature,
		Rules:          []string{"dont use external libraries"},
		// UseLexer removed
		UseAst: true, // Enable AST context by default
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
// Helper Functions (Only exporting where necessary)
// =============================================================================

// PrettyPrint prints colored text to the terminal. Exported for CLI use.
func PrettyPrint(color, text string) {
	fmt.Print(color, text, ColorReset)
}

// generatePrompt formats the final prompt for the LLM (internal helper).
func generatePrompt(contextPreamble, codeSnippet string, config Config) string {
	// Truncate code snippet if necessary
	if len(codeSnippet) > maxContentLength {
		// Truncate from the beginning to keep the end context
		codeSnippet = "..." + codeSnippet[len(codeSnippet)-maxContentLength+3:]
		log.Printf("Truncated code snippet to approximately %d characters.", maxContentLength)
	}

	// Truncate context preamble if it's excessively long (e.g., huge comments/scope)
	maxPreambleLen := 1024 // Adjust as needed
	if len(contextPreamble) > maxPreambleLen {
		contextPreamble = contextPreamble[:maxPreambleLen] + "\n... (context truncated)"
		log.Printf("Truncated context preamble to %d characters.", maxPreambleLen)
	}

	// Use the updated prompt template format string
	return fmt.Sprintf(config.PromptTemplate, contextPreamble, codeSnippet)
}

// OllamaError defines a custom error for Ollama API issues (internal).
type OllamaError struct {
	Message string
	Status  int // HTTP status code
}

func (e *OllamaError) Error() string {
	return fmt.Sprintf("Ollama error: %s (Status: %d)", e.Message, e.Status)
}

// callOllamaAPI handles the HTTP request to the Ollama generate endpoint (internal).
func callOllamaAPI(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	u, err := url.Parse(config.OllamaURL + "/api/generate")
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL: %w", err)
	}

	// Prepare payload, ensuring options are correctly nested
	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"temperature": config.Temperature,
			"num_ctx":     4096, // Increase context window if model supports it
			"top_p":       0.9,
			"stop":        config.Stop, // Pass stop sequences via options
		},
		// "stop": config.Stop, // Some Ollama versions might expect stop here
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
	req.Header.Set("Accept", "application/x-ndjson") // Expect newline-delimited JSON

	// Use a client with a timeout
	client := &http.Client{Timeout: 60 * time.Second} // Add a timeout
	resp, err := client.Do(req)
	if err != nil {
		// Handle network errors explicitly
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("ollama request timed out: %w", err)
		}
		return nil, fmt.Errorf("error making HTTP request to Ollama: %w", err)
	}

	// Handle non-200 status codes
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := "(could not read error body)"
		if readErr == nil {
			bodyString = string(bodyBytes)
			// Try to unmarshal Ollama's error structure
			var ollamaErrResp struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(bodyBytes, &ollamaErrResp) == nil && ollamaErrResp.Error != "" {
				bodyString = ollamaErrResp.Error
			}
		}
		// Provide specific error messages for common statuses
		if resp.StatusCode == http.StatusNotFound {
			return nil, &OllamaError{Message: fmt.Sprintf("Ollama API endpoint not found or model '%s' not available: %s", config.Model, bodyString), Status: resp.StatusCode}
		}
		if resp.StatusCode == http.StatusServiceUnavailable || resp.StatusCode == http.StatusTooManyRequests {
			// These are potentially retryable
			return nil, &OllamaError{Message: fmt.Sprintf("Ollama service unavailable or busy: %s", bodyString), Status: resp.StatusCode}
		}
		return nil, &OllamaError{Message: fmt.Sprintf("Ollama API request failed: %s", bodyString), Status: resp.StatusCode}
	}

	return resp.Body, nil
}

// streamCompletion reads and processes the NDJSON stream from Ollama (internal).
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer) error {
	defer r.Close()
	reader := bufio.NewReader(r)

	for {
		// Check for context cancellation before reading
		select {
		case <-ctx.Done():
			log.Println("Context cancelled during streaming")
			return ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')
		if err != nil {
			// Handle EOF gracefully
			if err == io.EOF {
				// Process any remaining data before EOF
				if len(line) > 0 {
					if procErr := processLine(line, w); procErr != nil {
						log.Printf("Error processing final line before EOF: %v", procErr)
						// Return the processing error if it occurred on the last line
						return procErr
					}
				}
				return nil // Successful end of stream
			}
			// Handle other read errors (e.g., network issues during stream)
			log.Printf("Error reading from Ollama stream: %v", err)
			// Check if the error is due to context cancellation which might have happened mid-read
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				return fmt.Errorf("error reading from Ollama stream: %w", err)
			}
		}

		// Process the line read
		if procErr := processLine(line, w); procErr != nil {
			log.Printf("Error processing Ollama response line: %v", procErr)
			// Decide if the error is fatal for the stream
			return procErr // Return the error to stop streaming
		}
	}
}

// processLine unmarshals and handles a single line from the Ollama stream (internal).
func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil // Skip empty lines
	}

	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		// Log malformed JSON but don't necessarily kill the stream
		log.Printf("Error unmarshalling Ollama JSON line: %v, line content: '%s'", err, string(line))
		return nil // Continue processing next line
	}

	// Check for errors reported within the JSON payload itself
	if resp.Error != "" {
		log.Printf("Ollama reported an error in stream: %s", resp.Error)
		return fmt.Errorf("ollama stream error: %s", resp.Error) // Fatal error from Ollama
	}

	// Write the actual response content to the output writer
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		log.Printf("Error writing completion chunk to output: %v", err)
		return fmt.Errorf("error writing to output: %w", err) // Fatal write error
	}

	// resp.Done is informational; EOF handling in streamCompletion is more reliable
	return nil
}

// retry implements a retry mechanism with backoff for specific errors (internal).
// Used only by GetCompletionStreamFromFile now.
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration) error {
	var err error
	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		err = operation()
		if err == nil {
			return nil // Success
		}

		// Check if the error is retryable
		var ollamaErr *OllamaError
		isRetryable := errors.As(err, &ollamaErr) && (ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		// TODO: Consider retrying on temporary network errors as well

		if !isRetryable {
			log.Printf("Non-retryable error encountered: %v", err)
			return err // Return non-retryable errors immediately
		}

		log.Printf("Attempt %d failed with retryable error: %v. Retrying in %v...", i+1, err, currentDelay)

		// Wait for the delay or context cancellation
		select {
		case <-ctx.Done():
			log.Printf("Context cancelled during retry wait: %v", ctx.Err())
			return ctx.Err() // Respect context cancellation
		case <-time.After(currentDelay):
			// Optional: Implement exponential backoff
			// currentDelay *= 2
		}
	}
	// If loop completes, all retries failed
	log.Printf("Operation failed after %d retries.", maxRetries)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, err) // Return the last error
}

// ProcessCodeWithMiddleware applies middleware functions sequentially. Exported.
func ProcessCodeWithMiddleware(codeSnippet string, middleware ...Middleware) (string, error) {
	currentCode := codeSnippet
	for i, m := range middleware {
		output, err := m(currentCode)
		if err != nil {
			return "", fmt.Errorf("middleware %d failed: %w", i+1, err)
		}
		currentCode = output
	}
	return currentCode, nil
}

// =============================================================================
// AST Analysis & Context Extraction (Internal Helpers)
// =============================================================================

// AnalyzeCodeContext parses the file, performs type checking, and extracts context (internal).
// Returns AstContextInfo and a combined error of all non-fatal analysis issues.
func AnalyzeCodeContext(ctx context.Context, filename string, line, col int) (info *AstContextInfo, analysisErr error) {
	// Initialize context info struct
	info = &AstContextInfo{
		FilePath:         filename, // Store original path initially
		VariablesInScope: make(map[string]types.Object),
		AnalysisErrors:   make([]error, 0), // Store non-fatal errors here
		CallArgIndex:     -1,               // Default to -1 (not in call args)
	}

	// Add panic recovery
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Panic recovered during AnalyzeCodeContext: %v\n%s", r, string(debug.Stack()))
			// Convert panic to error and add to analysis errors
			panicErr := fmt.Errorf("internal panic during analysis: %v", r)
			info.AnalysisErrors = append(info.AnalysisErrors, panicErr)
			// Set the main return error if it's not already set
			if analysisErr == nil {
				analysisErr = panicErr
			} else {
				// Use errors.Join (Go 1.20+) or wrap otherwise
				analysisErr = errors.Join(analysisErr, panicErr)
				// analysisErr = fmt.Errorf("%w; %w", analysisErr, panicErr) // Go < 1.20
			}
		}
	}()

	absFilename, err := filepath.Abs(filename)
	if err != nil {
		// This is a fatal setup error, return immediately
		return info, fmt.Errorf("failed to get absolute path for '%s': %w", filename, err)
	}
	info.FilePath = absFilename // Update with absolute path
	dir := filepath.Dir(absFilename)

	log.Printf("Starting context analysis for: %s (%d:%d)", absFilename, line, col)

	// --- 1. Load Package Information using go/packages ---
	fset := token.NewFileSet() // Create a FileSet for this analysis pass
	cfg := &packages.Config{
		Context: ctx,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedSyntax |
			packages.NeedTypesInfo | packages.NeedTypesSizes | packages.NeedDeps,
		Dir:  dir,
		Fset: fset,
		// Provide a custom ParseFile to handle potential syntax errors gracefully
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			// Try to parse comments and tolerate errors
			const mode = parser.ParseComments | parser.AllErrors
			file, err := parser.ParseFile(fset, filename, src, mode)
			// Don't return the error immediately, allow partial ASTs
			if err != nil {
				log.Printf("Parser error in %s (ignored for partial AST): %v", filename, err)
				// Collect parser errors if needed, but return the potentially partial file
			}
			return file, nil // Return file even if err is not nil
		},
		Tests: true, // Include test files
	}

	pkgs, loadErr := packages.Load(cfg, ".") // Load packages in the target directory
	if loadErr != nil {
		log.Printf("Error loading packages: %v", loadErr)
		// Store as non-fatal, attempt direct parse later
		info.AnalysisErrors = append(info.AnalysisErrors, fmt.Errorf("package loading failed: %w", loadErr))
	}
	// Log detailed errors from package loading if any occurred
	if packages.PrintErrors(pkgs) > 0 {
		log.Println("Detailed errors encountered during package loading (see above)")
		info.AnalysisErrors = append(info.AnalysisErrors, errors.New("package loading reported errors"))
	}

	// --- Find the target package, AST, and token.File ---
	var targetPkg *packages.Package
	var targetFileAST *ast.File
	var targetFile *token.File

	// Find the specific package and file AST we are interested in.
	for _, pkg := range pkgs {
		// Defensive check for nil fields which can happen if loading partially fails
		if pkg == nil || pkg.Fset == nil || pkg.Syntax == nil || pkg.CompiledGoFiles == nil {
			log.Printf("Skipping partially loaded/failed package: %s", pkg.ID)
			continue
		}
		// Iterate through the files associated with this package
		for i, filePath := range pkg.CompiledGoFiles {
			// Ensure index is valid for Syntax slice
			if i >= len(pkg.Syntax) {
				log.Printf("Mismatch between CompiledGoFiles and Syntax slices for package %s", pkg.ID)
				continue
			}
			astNode := pkg.Syntax[i]
			if astNode == nil {
				log.Printf("Nil AST node found for file %s in package %s", filePath, pkg.ID)
				continue
			}
			// Get the token.File for position mapping
			// Use the FileSet associated with the package!
			file := pkg.Fset.File(astNode.Pos())
			if file != nil && file.Name() == absFilename {
				// Found the file we are analyzing!
				targetPkg = pkg
				targetFileAST = astNode
				targetFile = file
				log.Printf("Found target AST via packages.Load: %s in package %s", filepath.Base(absFilename), targetPkg.Name)
				break
			}
		}
		if targetFileAST != nil {
			break // Stop searching packages once found
		}
	}

	// --- Fallback: Direct Parsing if packages.Load failed or didn't find the AST ---
	if targetFileAST == nil {
		log.Printf("Target file AST not found via packages.Load, attempting direct parse of %s", absFilename)
		srcBytes, readErr := os.ReadFile(absFilename)
		if readErr != nil {
			err := fmt.Errorf("target file AST not found and direct read failed for '%s': %w", absFilename, readErr)
			info.AnalysisErrors = append(info.AnalysisErrors, err)
			// Cannot proceed without source or AST, return combined errors
			analysisErr = errors.Join(info.AnalysisErrors...) // Update return error
			return info, analysisErr
		}
		// Use the same FileSet and parsing mode
		const mode = parser.ParseComments | parser.AllErrors
		var parseErr error
		// Use the FileSet created earlier (fset)
		targetFileAST, parseErr = parser.ParseFile(fset, absFilename, srcBytes, mode)
		if parseErr != nil {
			// Log parsing errors but continue if a partial AST was returned
			log.Printf("Direct parsing of %s failed: %v", absFilename, parseErr)
			info.AnalysisErrors = append(info.AnalysisErrors, fmt.Errorf("direct parsing failed: %w", parseErr))
		}
		if targetFileAST == nil {
			err := errors.New("failed to obtain any AST for target file")
			info.AnalysisErrors = append(info.AnalysisErrors, err)
			analysisErr = errors.Join(info.AnalysisErrors...) // Update return error
			return info, analysisErr                          // Cannot proceed without AST
		}
		// Get the token.File for the directly parsed file using the same fset
		targetFile = fset.File(targetFileAST.Pos())
		if targetFileAST.Name != nil {
			info.PackageName = targetFileAST.Name.Name // Get package name from AST
		} else {
			info.PackageName = "main" // Assume main if package name missing
			log.Println("Warning: Package name missing from directly parsed AST, assuming 'main'")
		}
		log.Printf("Using AST from direct parse. Package: %s", info.PackageName)
		// Note: Type information (targetPkg) will be unavailable in this fallback path
		targetPkg = nil // Explicitly nil out targetPkg
	} else if targetPkg != nil {
		// Defensive check: Ensure TypesInfo is populated if targetPkg is not nil
		if targetPkg.TypesInfo == nil {
			log.Println("Warning: targetPkg found but TypesInfo is nil. Type checking might have failed.")
			info.AnalysisErrors = append(info.AnalysisErrors, errors.New("package type info missing despite package load"))
			// Proceed without type info
		}
		info.PackageName = targetPkg.Name
	} else {
		// Defensive check: Should not happen if targetFileAST was found via packages.Load
		log.Println("Warning: targetFileAST found but targetPkg is nil")
		if targetFileAST.Name != nil {
			info.PackageName = targetFileAST.Name.Name
		} else {
			info.PackageName = "main"
		}
	}

	// Populate Imports from the AST
	if targetFileAST != nil {
		info.Imports = targetFileAST.Imports
	}

	// --- 2. Calculate Cursor Position (token.Pos) ---
	if targetFile == nil {
		err := errors.New("failed to get token.File for position calculation")
		info.AnalysisErrors = append(info.AnalysisErrors, err)
		analysisErr = errors.Join(info.AnalysisErrors...) // Update return error
		return info, analysisErr                          // Cannot proceed without token.File
	}
	// Convert 1-based line/col to 0-based offset -> token.Pos
	cursorPos, posErr := calculateCursorPos(targetFile, line, col)
	if posErr != nil {
		info.AnalysisErrors = append(info.AnalysisErrors, posErr)
		analysisErr = errors.Join(info.AnalysisErrors...) // Update return error
		return info, analysisErr                          // Cannot proceed without valid position
	}
	info.CursorPos = cursorPos
	log.Printf("Calculated cursor token.Pos: %d (Line: %d, Col: %d)", info.CursorPos, fset.Position(info.CursorPos).Line, fset.Position(info.CursorPos).Column)

	// --- 3. Find AST Path to Cursor and Identifier/Selector/Call at Cursor ---
	path, exact := astutil.PathEnclosingInterval(targetFileAST, info.CursorPos, info.CursorPos)
	if path == nil {
		err := errors.New("failed to find AST path enclosing cursor")
		info.AnalysisErrors = append(info.AnalysisErrors, err)
		// Might still be able to provide package-level context, don't return yet
		log.Println(err)
	} else {
		// Try to find the specific identifier, selector expression, or call expression at/before the cursor
		// Requires targetPkg for type resolution
		findContextNodes(path, info.CursorPos, targetPkg, info)
	}
	_ = exact // exact is true if the node covers the cursor position exactly

	// --- 4. Extract Context (Function, Scope) ---
	var preamble strings.Builder
	preamble.WriteString(fmt.Sprintf("// Context: File: %s, Package: %s\n", filepath.Base(absFilename), info.PackageName))

	// Extract info by walking *up* the AST path from the cursor
	if path != nil {
		for i := len(path) - 1; i >= 0; i-- {
			node := path[i]
			switch n := node.(type) {
			case *ast.FuncDecl:
				// Found enclosing function declaration
				if info.EnclosingFuncNode == nil { // Store the innermost one found
					info.EnclosingFuncNode = n
					funcName := ""
					if n.Name != nil {
						funcName = n.Name.Name
					}
					log.Printf("Found enclosing function node: %s", funcName)
					// Try to get type information if available (requires successful package load and non-nil targetPkg)
					// Defensive checks for targetPkg and its fields
					if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Defs != nil && n.Name != nil {
						if obj, ok := targetPkg.TypesInfo.Defs[n.Name]; ok && obj != nil {
							// Check if the object is indeed a function
							if fn, ok := obj.(*types.Func); ok {
								info.EnclosingFunc = fn
								preamble.WriteString(fmt.Sprintf("// Enclosing Function: %s\n", types.ObjectString(fn, types.RelativeTo(targetPkg.Types))))
								// Add parameters and named results to scope, check signature type defensively
								if sig, ok := fn.Type().(*types.Signature); ok {
									addSignatureToScope(sig, info.VariablesInScope)
								} else {
									log.Printf("Warning: Could not get signature for function %s", funcName)
								}
							} else {
								log.Printf("Type definition for %s is not *types.Func", funcName)
								preamble.WriteString(fmt.Sprintf("// Enclosing Function (AST): %s\n", formatFuncSignature(n)))
							}
						} else {
							log.Printf("No type definition found for function %s", funcName)
							preamble.WriteString(fmt.Sprintf("// Enclosing Function (AST): %s\n", formatFuncSignature(n)))
						}
					} else {
						preamble.WriteString(fmt.Sprintf("// Enclosing Function (AST): %s\n", formatFuncSignature(n)))
						if targetPkg == nil {
							log.Println("Type information unavailable for enclosing function (targetPkg is nil)")
						} else {
							log.Println("Type information unavailable for enclosing function (TypesInfo or Defs missing)")
						}
					}
				}

			case *ast.BlockStmt:
				// Found an enclosing block statement
				if info.EnclosingBlock == nil {
					info.EnclosingBlock = n
					log.Println("Found innermost enclosing block")
				}
				// Extract variables declared in this block *before* the cursor
				// Requires type info to be available
				// Defensive checks for targetPkg and its fields
				if targetPkg != nil && targetPkg.TypesInfo != nil && targetPkg.TypesInfo.Scopes != nil {
					scope := targetPkg.TypesInfo.Scopes[n]
					if scope != nil {
						addScopeVariables(scope, info.CursorPos, info.VariablesInScope)
					} else {
						log.Printf("No type scope found for block at pos %d", n.Pos())
					}
				} else {
					log.Println("Type/Scope information unavailable for block variables")
				}
				// Add cases for other relevant nodes like *ast.TypeSpec, *ast.File, etc. if needed
			}
		}
	} else {
		log.Println("AST path is nil, cannot determine enclosing function/block.")
	}

	// --- 5. Add Package Scope Variables ---
	// Defensive checks for targetPkg and its fields
	if targetPkg != nil && targetPkg.Types != nil {
		pkgScope := targetPkg.Types.Scope()
		if pkgScope != nil {
			log.Println("Adding package scope variables.")
			// Pass token.NoPos to include all package-level items regardless of cursor position
			addScopeVariables(pkgScope, token.NoPos, info.VariablesInScope)
		} else {
			log.Println("Package scope is nil.")
			info.AnalysisErrors = append(info.AnalysisErrors, errors.New("package scope information missing"))
		}
	} else {
		log.Println("Package type information unavailable, cannot add package scope.")
		// Don't add an error here if targetPkg was nil due to direct parsing fallback
		if targetPkg != nil { // Only add error if targetPkg existed but Types was nil
			info.AnalysisErrors = append(info.AnalysisErrors, errors.New("package type information missing"))
		}
	}

	// --- 6. Find Comments Near Cursor using CommentMap ---
	// Create CommentMap only if AST is valid
	var cmap ast.CommentMap
	if targetFileAST != nil {
		cmap = ast.NewCommentMap(fset, targetFileAST, targetFileAST.Comments)
	}
	// Pass the FileSet needed to interpret comment positions
	info.CommentsNearCursor = findCommentsWithMap(cmap, path, info.CursorPos, fset) // Use refined comment finding
	if len(info.CommentsNearCursor) > 0 {
		preamble.WriteString("// Relevant Comments:\n")
		for _, c := range info.CommentsNearCursor {
			// Sanitize comment slightly for prompt
			cleanComment := strings.TrimSpace(strings.TrimPrefix(c, "//"))
			cleanComment = strings.TrimSpace(strings.TrimPrefix(cleanComment, "/*"))
			cleanComment = strings.TrimSpace(strings.TrimSuffix(cleanComment, "*/"))
			preamble.WriteString(fmt.Sprintf("//   %s\n", cleanComment))
		}
	}

	// --- 7. Add Identifier/Selector/Call Info to Preamble ---
	qualifier := types.RelativeTo(targetPkg.Types) // Get qualifier once, handle nil pkg
	if info.CallExpr != nil {
		// Function call context
		funcName := "[unknown function]"
		// Try to get function name from AST
		switch fun := info.CallExpr.Fun.(type) {
		case *ast.Ident:
			funcName = fun.Name
		case *ast.SelectorExpr:
			if fun.Sel != nil {
				funcName = fun.Sel.Name
			}
		}
		preamble.WriteString(fmt.Sprintf("// Inside function call: %s (Arg %d)\n", funcName, info.CallArgIndex+1)) // Show 1-based arg index
		if info.CallExprFuncType != nil {
			preamble.WriteString(fmt.Sprintf("// Function Signature: %s\n", types.TypeString(info.CallExprFuncType, qualifier)))
		} else {
			preamble.WriteString("// Function Signature: (unknown)\n")
		}
	} else if info.SelectorExpr != nil && info.SelectorExprType != nil {
		// Selector expression context (e.g., x.y|)
		selName := ""
		if info.SelectorExpr.Sel != nil {
			selName = info.SelectorExpr.Sel.Name
		}
		typeName := types.TypeString(info.SelectorExprType, qualifier)
		preamble.WriteString(fmt.Sprintf("// Selector context: expr type = %s (selecting '%s')\n", typeName, selName))
		// List fields/methods if available
		members := listTypeMembers(info.SelectorExprType, targetPkg)
		if len(members) > 0 {
			preamble.WriteString("//   Available members:\n")
			sort.Strings(members) // Sort for consistency
			for _, member := range members {
				preamble.WriteString(fmt.Sprintf("//     - %s\n", member))
			}
		}

	} else if info.IdentifierAtCursor != nil {
		// Simple identifier context
		identName := info.IdentifierAtCursor.Name
		if info.IdentifierType != nil {
			typeName := types.TypeString(info.IdentifierType, qualifier) // Use TypeString for qualified name
			preamble.WriteString(fmt.Sprintf("// Identifier at cursor: %s (Type: %s)\n", identName, typeName))
		} else if info.IdentifierObject != nil {
			// Fallback if type is nil but object exists
			preamble.WriteString(fmt.Sprintf("// Identifier at cursor: %s (Object: %s)\n", identName, info.IdentifierObject.Name()))
		} else {
			preamble.WriteString(fmt.Sprintf("// Identifier at cursor: %s (Type unknown)\n", identName))
		}
	}

	// --- 8. Format Variables In Scope for Preamble ---
	if len(info.VariablesInScope) > 0 {
		preamble.WriteString("// Variables/Constants/Types in Scope:\n")
		var names []string
		for name := range info.VariablesInScope {
			names = append(names, name)
		}
		sort.Strings(names) // Sort for consistent order
		// Qualifier already determined above
		for _, name := range names {
			obj := info.VariablesInScope[name]
			// Use types.ObjectString for a readable representation including the type
			preamble.WriteString(fmt.Sprintf("//   %s\n", types.ObjectString(obj, qualifier)))
		}
	}

	// --- Finalize ---
	info.PromptPreamble = preamble.String()
	log.Printf("Generated Context Preamble:\n---\n%s\n---", info.PromptPreamble)

	logAnalysisErrors(info.AnalysisErrors) // Log any non-fatal errors

	// Return the info struct and a combined error (nil if no errors occurred)
	// Use errors.Join (Go 1.20+) to combine multiple errors cleanly
	analysisErr = errors.Join(info.AnalysisErrors...) // Update return error
	return info, analysisErr
}

// calculateCursorPos converts 1-based line/col to token.Pos (internal).
func calculateCursorPos(file *token.File, line, col int) (token.Pos, error) {
	if line <= 0 {
		return token.NoPos, fmt.Errorf("invalid line number: %d (must be >= 1)", line)
	}
	// Check if line exceeds file bounds early
	if line > file.LineCount() {
		return token.NoPos, fmt.Errorf("line number %d exceeds file line count %d", line, file.LineCount())
	}

	// Get the start offset of the target line
	lineStartOffset := file.LineStart(line)
	// This check should be redundant now due to LineCount check, but keep for safety
	if !lineStartOffset.IsValid() {
		// This case should ideally not be reached if line <= file.LineCount()
		return token.NoPos, fmt.Errorf("cannot get start offset for line %d in file '%s'", line, file.Name())
	}

	// col is 1-based byte column. Calculate 0-based offset from file start.
	// TODO: Handle multi-byte runes correctly if col represents rune column.
	// Assuming byte column for now.
	cursorOffset := int(lineStartOffset) + col - 1

	// Validate column within the line/file bounds. Allow position just after last char.
	maxOffset := file.Size()
	if cursorOffset < int(file.Pos(0)) || cursorOffset > maxOffset { // Check lower bound too
		log.Printf("Warning: column %d results in offset %d which is outside valid range [0, %d] for line %d. Clamping.", col, cursorOffset, maxOffset, line)
		if cursorOffset > maxOffset {
			cursorOffset = maxOffset // Clamp to end of file
		}
		if cursorOffset < int(file.Pos(0)) {
			cursorOffset = int(file.Pos(0)) // Clamp to start of file
		}
	}

	// Convert the final offset to token.Pos
	pos := file.Pos(cursorOffset)
	if !pos.IsValid() {
		// Should be valid if offset is within bounds, but double-check
		return token.NoPos, fmt.Errorf("failed to calculate valid token.Pos for offset %d", cursorOffset)
	}
	return pos, nil
}

// addSignatureToScope adds parameters and named results from a function signature to the scope map (internal).
func addSignatureToScope(sig *types.Signature, scope map[string]types.Object) {
	if sig == nil {
		return
	}
	// Add parameters
	params := sig.Params()
	if params != nil {
		for j := 0; j < params.Len(); j++ {
			param := params.At(j)
			if param != nil && param.Name() != "" { // Ensure param is valid and named
				if _, exists := scope[param.Name()]; !exists {
					scope[param.Name()] = param
				}
			}
		}
	}
	// Add named results
	results := sig.Results()
	if results != nil {
		for j := 0; j < results.Len(); j++ {
			res := results.At(j)
			// Only add named results that haven't been shadowed by params
			if res != nil && res.Name() != "" {
				if _, exists := scope[res.Name()]; !exists {
					scope[res.Name()] = res
				}
			}
		}
	}
}

// addScopeVariables adds variables from a types.Scope to the scope map (internal).
// Modified to optionally include all scope items regardless of position if cursorPos is NoPos.
func addScopeVariables(typeScope *types.Scope, cursorPos token.Pos, scopeMap map[string]types.Object) {
	if typeScope == nil {
		return
	}
	for _, name := range typeScope.Names() {
		obj := typeScope.Lookup(name)
		// Include if cursor position is invalid (e.g., package scope using token.NoPos)
		// OR object defined before cursor
		// OR object has no position (like builtins)
		include := !cursorPos.IsValid() || !obj.Pos().IsValid() || obj.Pos() < cursorPos

		if obj != nil && include {
			switch obj.(type) {
			case *types.Var, *types.Const, *types.TypeName, *types.Func, *types.Label, *types.PkgName, *types.Builtin: // Include more object types
				if _, exists := scopeMap[name]; !exists { // Add if not already shadowed by inner scope
					scopeMap[name] = obj
				}
			case *types.Nil: // Handle nil type specifically if needed
				if _, exists := scopeMap[name]; !exists {
					scopeMap[name] = obj
				}
			default:
				// This might be noisy, comment out unless debugging scope issues
				// log.Printf("Unhandled object type in scope: %T", obj)
				_ = obj // Avoid unused variable error if log is commented out
			}
		}
	}
}

// formatFuncSignature creates a string representation from an ast.FuncDecl (fallback, internal).
func formatFuncSignature(f *ast.FuncDecl) string {
	var sb strings.Builder
	sb.WriteString("func ")
	if f.Recv != nil && len(f.Recv.List) > 0 {
		// Basic receiver formatting - TODO: Improve this if needed
		sb.WriteString("(...) ")
	}
	if f.Name != nil {
		sb.WriteString(f.Name.Name)
	} else {
		sb.WriteString("[anonymous]")
	}
	// Basic parameter/result formatting - TODO: Improve this if needed
	sb.WriteString("(...)") // Assume params exist
	if f.Type != nil && f.Type.Results != nil {
		sb.WriteString(" (...)") // Indicate results exist
	}
	return sb.String()
}

// findCommentsWithMap uses ast.CommentMap to find comments near the cursor or associated with enclosing nodes (internal).
func findCommentsWithMap(cmap ast.CommentMap, path []ast.Node, cursorPos token.Pos, fset *token.FileSet) []string {
	var comments []string
	// Defend against nil inputs
	if cmap == nil || !cursorPos.IsValid() || fset == nil {
		log.Println("Skipping comment analysis due to nil CommentMap, FileSet, or invalid cursor position.")
		return comments
	}

	cursorLine := fset.Position(cursorPos).Line

	// 1. Check comments immediately preceding the cursor line
	// Iterate through all comments mapped to nodes (more reliable than iterating all comments)
	foundPreceding := false
	var precedingComments []string // Store preceding comments separately first
	for node := range cmap {       // Iterate through nodes that have comments
		if node == nil {
			continue
		}
		// Optimization: If node starts after cursor, skip? Not reliable for preceding comments.

		for _, cg := range cmap[node] { // Get comment groups for this node
			if cg == nil {
				continue
			}
			commentEndLine := fset.Position(cg.End()).Line
			// Check if comment group ends exactly on the line before the cursor
			if commentEndLine == cursorLine-1 {
				log.Printf("Found preceding comment group (line %d) associated with node %T", commentEndLine, node)
				for _, c := range cg.List {
					if c != nil {
						precedingComments = append(precedingComments, c.Text)
					}
				}
				foundPreceding = true
				break // Found comments for this node, move to next node
			}
		}
		if foundPreceding {
			break
		} // Prioritize the first preceding group found
	}
	// If preceding comment found, return it immediately as most relevant
	if foundPreceding {
		comments = append(comments, precedingComments...)
		// Remove duplicates just in case (should be rare here)
		seen := make(map[string]struct{})
		uniqueComments := make([]string, 0, len(comments))
		for _, c := range comments {
			if _, ok := seen[c]; !ok {
				seen[c] = struct{}{}
				uniqueComments = append(uniqueComments, c)
			}
		}
		log.Printf("Using preceding comments: %d lines", len(uniqueComments))
		return uniqueComments
	}

	// 2. If no preceding comment, check Doc comments associated with the enclosing nodes in the path
	if path != nil {
		// Check innermost node first, then walk up
		for i := 0; i < len(path); i++ {
			node := path[i] // Start from innermost node
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
				docComment = n.Doc // Doc for var/const in GenDecl
				// Add other cases if needed
			}
			if docComment != nil {
				log.Printf("Found Doc comment for enclosing node %T", node)
				for _, c := range docComment.List {
					if c != nil {
						comments = append(comments, c.Text)
					}
				}
				// Found Doc comment, prioritize this and stop searching further up
				goto cleanup // Use goto to jump to cleanup/deduplication
			}
		}
	}

	// 3. If no preceding or Doc comment, check general comments attached by CommentMap to enclosing nodes
	if path != nil {
		// Check innermost node first, then walk up
		for i := 0; i < len(path); i++ {
			node := path[i] // Start from innermost node
			// Check general comments attached by CommentMap
			if cgs := cmap.Filter(node).Comments(); len(cgs) > 0 {
				log.Printf("Found general comments associated with enclosing node %T at pos %d", node, node.Pos())
				for _, cg := range cgs {
					for _, c := range cg.List {
						if c != nil {
							comments = append(comments, c.Text)
						}
					}
				}
				// Found comments for an enclosing node, break after first one found walking up.
				break
			}
		}
	}

cleanup:
	// Remove duplicates from collected comments (if multiple sources added them)
	seen := make(map[string]struct{})
	uniqueComments := make([]string, 0, len(comments))
	for _, c := range comments {
		if _, ok := seen[c]; !ok {
			seen[c] = struct{}{}
			uniqueComments = append(uniqueComments, c)
		}
	}
	if len(uniqueComments) > 0 {
		log.Printf("Using associated comments: %d lines", len(uniqueComments))
	} else {
		log.Println("No relevant comments found near cursor.")
	}

	return uniqueComments
}

// findContextNodes tries to find the identifier, selector expression, or call expression
// at/before the cursor and resolve its type information, updating the AstContextInfo struct directly.
func findContextNodes(path []ast.Node, cursorPos token.Pos, pkg *packages.Package, info *AstContextInfo) {
	// Defensive check for type info
	if len(path) == 0 {
		log.Println("Cannot find context nodes: Missing AST path.")
		return
	}
	// Type info might be missing if package loading/checking failed
	hasTypeInfo := pkg != nil && pkg.TypesInfo != nil

	// Check innermost nodes first for specific patterns like CallExpr or SelectorExpr
	innermostNode := path[0]

	// --- Check for Call Expression ---
	if callExpr, ok := innermostNode.(*ast.CallExpr); ok {
		// Cursor is likely inside the arguments of a call expression
		// Check if cursor is within the parens `()`
		if cursorPos > callExpr.Lparen && cursorPos <= callExpr.Rparen {
			log.Printf("Cursor is inside CallExpr at pos %d", callExpr.Pos())
			info.CallExpr = callExpr
			// Determine argument index
			info.CallArgIndex = 0 // Default to first arg
			for i, arg := range callExpr.Args {
				if cursorPos > arg.End() { // If cursor is after this arg, it's potentially the next one
					info.CallArgIndex = i + 1
				} else {
					break // Cursor is before or within this arg
				}
			}
			log.Printf("Cursor likely at argument index %d", info.CallArgIndex)

			// Try to resolve the type of the function being called
			if hasTypeInfo {
				// Defensive check for Types map
				if pkg.TypesInfo.Types == nil {
					log.Println("Warning: TypesInfo.Types map is nil, cannot resolve call expr type.")
					info.AnalysisErrors = append(info.AnalysisErrors, errors.New("types map missing for call expr"))
					return // Cannot proceed with type resolution here
				}
				if tv, ok := pkg.TypesInfo.Types[callExpr.Fun]; ok && tv.IsValue() {
					// Check if the type is a signature
					// Defensive type assertion
					if sig, ok := tv.Type.Underlying().(*types.Signature); ok { // Use Underlying() for robustness
						info.CallExprFuncType = sig
						log.Printf("Resolved call expression function type: %s", types.TypeString(sig, types.RelativeTo(pkg.Types)))
					} else {
						log.Printf("Warning: Type of call expression function is not a signature: %T", tv.Type)
						info.AnalysisErrors = append(info.AnalysisErrors, fmt.Errorf("type of call func is %T, not signature", tv.Type))
					}
				} else {
					log.Printf("Could not resolve type of function in call expression: %T", callExpr.Fun)
					info.AnalysisErrors = append(info.AnalysisErrors, fmt.Errorf("missing type info for call func %T", callExpr.Fun))
				}
			} else {
				log.Println("Skipping call expr type resolution due to missing type info.")
			}
			return // Prioritize CallExpr context
		}
	}

	// --- Check for Selector Expression ---
	// Check if the innermost node OR the node just before it is a SelectorExpr
	for i := 0; i < len(path) && i < 2; i++ { // Check first 2 nodes in path
		if selExpr, ok := path[i].(*ast.SelectorExpr); ok {
			// Cursor is likely after the dot `.`
			// Check if cursor is after the base expression `X` and potentially at/after the selector `Sel`
			if cursorPos > selExpr.X.End() {
				log.Printf("Cursor is potentially within or after selector expression: %s", selExpr.Sel.Name)
				info.SelectorExpr = selExpr
				// Get the type of the expression being selected *from* (X)
				if hasTypeInfo {
					// Defensive check for Types map
					if pkg.TypesInfo.Types == nil {
						log.Println("Warning: TypesInfo.Types map is nil, cannot resolve selector expr type.")
						info.AnalysisErrors = append(info.AnalysisErrors, errors.New("types map missing for selector expr"))
						return // Cannot proceed with type resolution here
					}
					if typeAndValue, ok := pkg.TypesInfo.Types[selExpr.X]; ok {
						info.SelectorExprType = typeAndValue.Type
						if info.SelectorExprType != nil {
							log.Printf("Selector base expression type: %s", info.SelectorExprType.String())
						} else {
							log.Println("Could not determine type of selector base expression.")
							info.AnalysisErrors = append(info.AnalysisErrors, fmt.Errorf("missing type for selector base %T", selExpr.X))
						}
					} else {
						log.Println("No type info found for selector base expression.")
						info.AnalysisErrors = append(info.AnalysisErrors, fmt.Errorf("missing type info for selector base %T", selExpr.X))
					}
				} else {
					log.Println("Skipping selector expr type resolution due to missing type info.")
				}
				// If we found a selector, prioritize this context
				return
			}
		}
	}

	// --- Check for Simple Identifier ---
	// If not a call or selector, look for the most specific *identifier* ending at/before cursor
	var ident *ast.Ident
	for _, node := range path {
		if id, ok := node.(*ast.Ident); ok {
			// Check if cursor is within or immediately after the identifier
			// Use a slightly more lenient check: identifier ends near the cursor
			// CursorPos is the position *before* which we want to complete.
			// So, check if the identifier ends exactly at the cursor position.
			if id.End() == cursorPos { // || (id.Pos() <= cursorPos && cursorPos <= id.End()) { // Check if cursor is *within* ident? Less common for completion.
				ident = id
				// Don't break immediately, allow finding a more specific ident deeper in path? Or break? Let's break.
				break
			}
		}
	}

	if ident == nil {
		log.Println("No specific identifier found ending at cursor position for type analysis.")
		return
	}

	info.IdentifierAtCursor = ident
	log.Printf("Found potential identifier '%s' ending at pos %d (cursor %d)", ident.Name, ident.End(), cursorPos)

	// Now try to get the type info for this identifier (requires pkg info)
	if !hasTypeInfo {
		log.Println("Skipping identifier type resolution due to missing package/type info.")
		return
	}

	// Defensive checks for maps within TypesInfo
	if pkg.TypesInfo.Uses == nil && pkg.TypesInfo.Defs == nil && pkg.TypesInfo.Types == nil {
		log.Println("Warning: TypesInfo maps (Uses, Defs, Types) are nil. Cannot resolve identifier type.")
		info.AnalysisErrors = append(info.AnalysisErrors, errors.New("type info maps (Uses, Defs, Types) are nil"))
		return
	}

	var obj types.Object
	var typ types.Type

	// Check Uses first (refers to an existing object)
	if pkg.TypesInfo.Uses != nil {
		if usesObj := pkg.TypesInfo.Uses[ident]; usesObj != nil {
			obj = usesObj
			typ = obj.Type() // Get type from the object it uses
			log.Printf("Identifier '%s' uses object: %s (Type: %s)", ident.Name, obj.Name(), typ.String())
		}
	}

	// Check Defs if not found in Uses
	if obj == nil && pkg.TypesInfo.Defs != nil {
		if defObj := pkg.TypesInfo.Defs[ident]; defObj != nil {
			obj = defObj
			typ = obj.Type() // Get type from the defined object
			log.Printf("Identifier '%s' defines object: %s (Type: %s)", ident.Name, obj.Name(), typ.String())
		}
	}

	// Check Implicits if still not found
	if obj == nil && pkg.TypesInfo.Implicits != nil {
		if impObj := pkg.TypesInfo.Implicits[ident]; impObj != nil {
			obj = impObj
			typ = obj.Type()
			log.Printf("Identifier '%s' is implicit object: %s (Type: %s)", ident.Name, obj.Name(), typ.String())
		}
	}

	// Fallback: try TypeOf if no object association found
	if obj == nil && pkg.TypesInfo.Types != nil {
		if tv, ok := pkg.TypesInfo.Types[ident]; ok && tv.Type != nil {
			typ = tv.Type
			log.Printf("Identifier '%s' has type (TypeOf fallback): %s", ident.Name, typ.String())
		} else {
			log.Printf("Could not determine object or type for identifier '%s'", ident.Name)
			info.AnalysisErrors = append(info.AnalysisErrors, fmt.Errorf("missing type info for identifier '%s'", ident.Name))
		}
	}

	// Defensive check: If we found an object but type is nil, log it
	if obj != nil && typ == nil {
		log.Printf("Warning: Found object '%s' for identifier '%s', but its type is nil.", obj.Name(), ident.Name)
		info.AnalysisErrors = append(info.AnalysisErrors, fmt.Errorf("object '%s' found but type is nil", obj.Name()))
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
	qualifier := types.RelativeTo(pkg.Types) // For formatting types

	// Dereference pointer types to get the underlying struct/interface
	if ptr, ok := typ.Underlying().(*types.Pointer); ok {
		typ = ptr.Elem()
	}

	// Handle named types (most common case for structs/interfaces with methods)
	if named, ok := typ.(*types.Named); ok {
		// List methods
		for i := 0; i < named.NumMethods(); i++ {
			method := named.Method(i)
			if method.Exported() { // Only list exported members
				members = append(members, types.ObjectString(method, qualifier))
			}
		}
		// If it's a named struct, list fields
		if underlyingStruct, ok := named.Underlying().(*types.Struct); ok {
			for i := 0; i < underlyingStruct.NumFields(); i++ {
				field := underlyingStruct.Field(i)
				if field.Exported() { // Only list exported members
					members = append(members, types.ObjectString(field, qualifier))
				}
			}
		}
		// TODO: Handle named interfaces similarly?
	} else if iface, ok := typ.Underlying().(*types.Interface); ok {
		// List methods for interface type
		for i := 0; i < iface.NumExplicitMethods(); i++ {
			method := iface.ExplicitMethod(i)
			if method.Exported() {
				members = append(members, types.ObjectString(method, qualifier))
			}
		}
		// Also consider embedded interfaces
		for i := 0; i < iface.NumEmbeddeds(); i++ {
			embeddedType := iface.EmbeddedType(i)
			// Recursively list members of embedded type? Careful about cycles.
			// For simplicity, just indicate embedding for now.
			members = append(members, fmt.Sprintf("// embeds %s", types.TypeString(embeddedType, qualifier)))
		}
	} else if st, ok := typ.Underlying().(*types.Struct); ok {
		// Handle anonymous structs directly
		for i := 0; i < st.NumFields(); i++ {
			field := st.Field(i)
			if field.Exported() { // Check if field is exported (relevant for embedded structs)
				members = append(members, types.ObjectString(field, qualifier))
			}
		}
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

// =============================================================================
// Main Completion Functions (Exported API)
// =============================================================================

// GetCompletion provides completion for a raw code snippet (no file context). Non-streaming. Exported.
// Simplified: Removed retry logic for basic prompt completion.
func GetCompletion(ctx context.Context, codeSnippet string, config Config) (string, error) {
	log.Println("GetCompletion called for basic prompt.")
	contextPreamble := "// Provide Go code completion below."
	prompt := generatePrompt(contextPreamble, codeSnippet, config)

	// Single attempt, no retry for basic prompt
	reader, err := callOllamaAPI(ctx, prompt, config)
	if err != nil {
		// Don't wrap retryable errors here, just return the error
		return "", fmt.Errorf("failed to call Ollama API: %w", err)
	}

	var buffer bytes.Buffer
	// Use a child context with timeout for the streaming part itself
	streamCtx, cancelStream := context.WithTimeout(ctx, 50*time.Second) // Shorter timeout for stream processing
	defer cancelStream()
	if streamErr := streamCompletion(streamCtx, reader, &buffer); streamErr != nil {
		// Don't retry on stream errors unless specifically identifiable as temporary
		return "", fmt.Errorf("error during stream processing: %w", streamErr) // Make stream errors non-retryable by default
	}
	ollamaResponse := buffer.String()

	return ollamaResponse, nil
}

// GetCompletionStreamFromFile is the primary function using AST context and streaming output. Exported.
func GetCompletionStreamFromFile(ctx context.Context, filename string, row, col int, config Config, w io.Writer) error {
	var contextPreamble string = "// Basic file context only.\n" // Default preamble
	var analysisErr error                                        // Stores combined error from analysis

	// --- 1. Analyze Context (AST/Types if enabled) ---
	if config.UseAst {
		log.Printf("Analyzing context using AST/Types for %s:%d:%d", filename, row, col)
		analysisCtx, cancelAnalysis := context.WithTimeout(ctx, 20*time.Second) // Timeout for analysis
		defer cancelAnalysis()

		var analysisInfo *AstContextInfo
		// Assign combined error directly to analysisErr
		analysisInfo, analysisErr = AnalyzeCodeContext(analysisCtx, filename, row, col)

		// Check if analysis produced usable info, even if non-fatal errors occurred
		if analysisInfo != nil && analysisInfo.PromptPreamble != "" {
			contextPreamble = analysisInfo.PromptPreamble // Use generated preamble
			log.Println("Successfully generated AST-based context preamble.")
		} else {
			// Analysis failed to produce preamble or info was nil
			log.Println("AST analysis returned no specific context preamble.")
			contextPreamble += "// Warning: AST analysis returned no specific context.\n"
		}
		// Log analysis errors regardless of whether preamble was generated
		if analysisErr != nil {
			// Use the helper to log potentially combined errors
			logAnalysisErrors([]error{analysisErr}) // Wrap in slice for helper
			// Optionally append warning to preamble if errors occurred?
			// contextPreamble += fmt.Sprintf("// Warning: Analysis errors: %v\n", analysisErr)
		}
	} else {
		log.Println("AST analysis disabled, using basic file context.")
	}

	// --- 2. Extract Code Snippet for Prompt ---
	codeSnippet, err := extractCodeSnippet(filename, row, col)
	if err != nil {
		return fmt.Errorf("failed to extract code snippet: %w", err)
	}

	// --- 3. Prepare Prompt and Call Ollama ---
	prompt := generatePrompt(contextPreamble, codeSnippet, config)
	log.Printf("Generated Prompt (length %d):\n---\n%s\n---", len(prompt), prompt) // Log for debugging

	// Use retry mechanism only for the streaming call
	apiCallFunc := func() error {
		apiCtx, cancelApi := context.WithTimeout(ctx, 60*time.Second) // Timeout for API call + streaming
		defer cancelApi()

		reader, apiErr := callOllamaAPI(apiCtx, prompt, config)
		if apiErr != nil {
			// If API call fails, it might be due to bad context from analysis errors.
			// Log both if analysis errors existed.
			if analysisErr != nil {
				log.Printf("Reporting API error (%v) which might be related to prior analysis errors (%v)", apiErr, analysisErr)
			}
			// Let retry logic handle retryable OllamaErrors
			return apiErr // Return error for retry check
		}

		// --- 4. Stream Completion ---
		fmt.Fprintf(w, "%sCompletion:%s\n", ColorGreen, ColorReset) // Indicate start of completion output
		streamErr := streamCompletion(apiCtx, reader, w)            // Pass the API context, rename err
		fmt.Fprintln(w)                                             // Add a final newline

		if streamErr != nil {
			// Check for context errors vs other stream errors
			if errors.Is(streamErr, context.Canceled) {
				log.Println("Completion stream explicitly cancelled.")
				// Return non-retryable error
				return fmt.Errorf("completion stream cancelled: %w", streamErr)
			}
			if errors.Is(streamErr, context.DeadlineExceeded) {
				log.Println("Completion stream timed out.")
				// Return non-retryable error
				return fmt.Errorf("completion stream timed out: %w", streamErr)
			}
			// Other stream processing errors - assume non-retryable for now
			return fmt.Errorf("error during completion streaming: %w", streamErr)
		}
		// If streaming succeeded
		log.Println("Completion stream finished successfully for this attempt.")
		return nil // Success for this attempt
	}

	// Execute the API call with retry logic
	err = retry(ctx, apiCallFunc, maxRetries, retryDelay)

	if err != nil {
		// If retries failed, return the final error
		return fmt.Errorf("failed to get completion stream after retries: %w", err)
	}

	// Report analysis errors as a final warning even if streaming succeeded?
	if analysisErr != nil {
		log.Printf("Warning: Completion succeeded, but context analysis encountered errors: %v", analysisErr)
		// Optionally return analysisErr here if desired?
		// return analysisErr
	}
	return nil // Overall success
}

// =============================================================================
// Middleware Definitions (Exported Examples)
// =============================================================================

// LintResult checks for balanced braces (simple example). Exported.
func LintResult(input string) (string, error) {
	openBraces := strings.Count(input, "{")
	closeBraces := strings.Count(input, "}")
	if openBraces > closeBraces {
		log.Printf("Lint warning: potentially unbalanced braces (open > close)")
	} else if closeBraces > openBraces {
		log.Printf("Lint warning: potentially unbalanced braces (close > open)")
		// return "", fmt.Errorf("lint error: unmatched closing braces '}'") // Make it an error if desired
	}
	// Add more sophisticated linting if needed (e.g., using go/parser)
	return input, nil
}

// RemoveExternalLibraries attempts to filter import lines (basic example). Exported.
func RemoveExternalLibraries(input string) (string, error) {
	// This is very basic and might remove valid code or miss aliased imports.
	// A proper implementation would parse the import block using go/ast.
	lines := strings.Split(input, "\n")
	filteredLines := make([]string, 0, len(lines))
	inImportBlock := false
	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "import (") {
			inImportBlock = true
			// Keep the import block start for structure, or filter block entirely?
			// Let's keep it for now.
			filteredLines = append(filteredLines, line)
			continue
		}
		if inImportBlock && strings.HasPrefix(trimmedLine, ")") {
			inImportBlock = false
			filteredLines = append(filteredLines, line)
			continue
		}

		isImportLine := strings.HasPrefix(trimmedLine, `import `) || (inImportBlock && strings.Contains(trimmedLine, `"`))

		if isImportLine {
			// Basic check for likely standard library paths
			isStdLib := strings.Contains(trimmedLine, `"fmt"`) || strings.Contains(trimmedLine, `"os"`) ||
				strings.Contains(trimmedLine, `"strings"`) || strings.Contains(trimmedLine, `"errors"`) ||
				strings.Contains(trimmedLine, `"log"`) || strings.Contains(trimmedLine, `"context"`) ||
				strings.Contains(trimmedLine, `"time"`) || strings.Contains(trimmedLine, `"net"`) || // Broader net check
				strings.Contains(trimmedLine, `"encoding/"`) || strings.Contains(trimmedLine, `"sync"`) ||
				strings.Contains(trimmedLine, `"sort"`) || strings.Contains(trimmedLine, `"path/"`) ||
				strings.Contains(trimmedLine, `"go/"`) || strings.Contains(trimmedLine, `"bufio"`) ||
				strings.Contains(trimmedLine, `"bytes"`) || strings.Contains(trimmedLine, `"io"`) ||
				strings.Contains(trimmedLine, `"regexp"`) || strings.Contains(trimmedLine, `"strconv"`) ||
				strings.Contains(trimmedLine, `"runtime/"`) // Added runtime

			// Allow golang.org/x/ packages
			isGolangX := strings.Contains(trimmedLine, `"golang.org/x/`)

			if strings.Contains(trimmedLine, `"`) && !isStdLib && !isGolangX {
				log.Printf("Filtering potential external library import: %s", trimmedLine)
				continue // Skip this line
			}
		}
		filteredLines = append(filteredLines, line)
	}
	return strings.Join(filteredLines, "\n"), nil
}

// =============================================================================
// Spinner & File Helpers (Spinner Exported, others internal)
// =============================================================================

// Spinner for terminal feedback. Exported Type.
type Spinner struct {
	chars    []string
	index    int
	mu       sync.Mutex
	stopChan chan struct{}
	running  bool
}

// NewSpinner creates a new Spinner. Exported Function.
func NewSpinner() *Spinner {
	return &Spinner{
		chars:    []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}, // Extended chars
		index:    0,
		stopChan: make(chan struct{}), // Initialize channel
	}
}

// Start starts the spinner animation in a goroutine. Exported Method.
func (s *Spinner) Start(message string) {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return // Already running
	}
	// Ensure the stop channel is fresh if Start is called after Stop
	select {
	case <-s.stopChan: // If channel is closed, create a new one
	default:
	}
	s.stopChan = make(chan struct{})
	s.running = true
	s.mu.Unlock()

	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		defer func() {
			s.mu.Lock()
			s.running = false
			s.mu.Unlock()
			fmt.Print("\r\033[K") // Clear the line completely on stop
		}()

		for {
			select {
			case <-s.stopChan:
				return // Exit goroutine
			case <-ticker.C:
				s.mu.Lock()
				char := s.chars[s.index]
				s.index = (s.index + 1) % len(s.chars)
				// Use carriage return (\r) and clear line (\033[K)
				fmt.Printf("\r\033[K%s%s%s %s", ColorCyan, char, ColorReset, message)
				s.mu.Unlock()
			}
		}
	}()
}

// Stop stops the spinner animation. Exported Method.
func (s *Spinner) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.running {
		return // Not running
	}
	// Close channel safely
	select {
	case <-s.stopChan: // Already closed
	default:
		close(s.stopChan)
	}
	// The defer in the goroutine handles clearing the line.
}

// extractCodeSnippet reads file content up to the cursor position (internal).
func extractCodeSnippet(filename string, row, col int) (string, error) {
	// Read the entire file content
	contentBytes, err := os.ReadFile(filename)
	if err != nil {
		return "", fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	content := string(contentBytes)

	// We need to find the byte offset corresponding to row, col to insert the cursor marker
	// or determine the split point. We'll use the FileSet approach again, but just for offset calculation.
	fset := token.NewFileSet()
	// AddFile requires size, which we have. Use a dummy name if filename isn't absolute.
	absFilename, _ := filepath.Abs(filename)
	if absFilename == "" {
		absFilename = "input.go"
	}
	// Need to handle potential errors from AddFile if size is incorrect, though unlikely here
	file := fset.AddFile(absFilename, fset.Base(), len(contentBytes))
	if file == nil {
		return "", fmt.Errorf("failed to add file '%s' to fileset", absFilename)
	}

	// Calculate the token.Pos for the cursor
	cursorPos, posErr := calculateCursorPos(file, row, col)
	if posErr != nil {
		// If position is invalid, maybe just return the whole file content? Or error out?
		// Let's error out for now, as completion needs a valid point.
		return "", fmt.Errorf("cannot determine valid cursor position for snippet extraction: %w", posErr)
	}

	// Get the byte offset from the token.Pos
	// Defend against invalid pos before calling Offset
	if !cursorPos.IsValid() {
		return "", fmt.Errorf("invalid cursor position calculated before offset")
	}
	offset := file.Offset(cursorPos)

	// Defend against invalid offset, although calculateCursorPos should clamp it
	if offset < 0 || offset > len(content) {
		return "", fmt.Errorf("calculated offset %d is out of bounds [0, %d] for file size %d", offset, len(content), len(content))
	}

	// Return the content up to the cursor offset.
	// The LLM prompt structure now handles where the completion should go.
	return content[:offset], nil
}

// readComments is kept but less critical now as comments are handled via AST analysis (internal).
func readComments(code string) []string {
	var comments []string
	lines := strings.Split(code, "\n")
	inBlockComment := false
	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)

		// Handle block comments
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
			continue // Move to next line if it was start of block
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
			continue // Move to next line
		}

		// Handle single-line comments
		if strings.HasPrefix(trimmedLine, "//") {
			comment := strings.TrimSpace(strings.TrimPrefix(trimmedLine, "//"))
			if comment != "" {
				comments = append(comments, comment)
			}
		}
	}
	return comments
}
