// deepcomplete/deepcomplete.go
package deepcomplete

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors" // Import the errors package
	"fmt"
	"go/scanner"
	"go/token"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"

	// "os/exec" // No longer needed for lexer
	"strings"
	"sync"
	"time"
	// ollamago "github.com/ollama/ollama/api" // Optional: Import ollama-go library
)

// =============================================================================
// Constants
// =============================================================================

const (
	defaultOllamaURL = "http://localhost:11434"
	defaultModel     = "deepseek-coder-r2"
	// Corrected promptTemplate: Use standard string literal for escape sequences
	// and escape inner backticks if needed, or structure differently.
	// Using concatenation for clarity here.
	promptTemplate = `<s>[INST] <<SYS>>
You are an AI coding assistant. You are an expert Go programmer.
You are helping the user complete their code.
Only complete the code, do not add any comments.
Do not add any newlines or indentation.
<</SYS>>

Complete the following Go code:
` + "```go\n%s\n```" + `
[/INST]`
	maxRetries         = 3
	retryDelay         = 2 * time.Second
	maxContentLength   = 2048 // Limit content length to prevent excessively long prompts.
	defaultMaxTokens   = 256
	defaultStop        = "\n"
	defaultTemperature = 0.1
)

// =============================================================================
// Types
// =============================================================================

// Config holds the configuration for the autocompletion.
type Config struct {
	OllamaURL      string
	Model          string
	PromptTemplate string
	MaxTokens      int
	Stop           []string
	Temperature    float64
	Rules          []string // Custom rules.
	UseLexer       bool
}

// OllamaResponse represents the response structure for streaming.
// Note: The official ollama-go library provides its own types.
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"` // Include error field
}

// Middleware is a function that can process the input or output.
type Middleware func(input string) (string, error)

// LexerResult holds the result of lexing using go/scanner.
type LexerResult struct {
	Tokens    []string      // The token literals
	Positions []token.Pos   // The token positions
	Kinds     []token.Token // The token kinds (e.g., token.IDENT, token.INT)
}

// =============================================================================
// Variables
// =============================================================================

var (
	// DefaultConfig is the default configuration.
	DefaultConfig = Config{
		OllamaURL:      defaultOllamaURL,
		Model:          defaultModel,
		PromptTemplate: promptTemplate,
		MaxTokens:      defaultMaxTokens,
		Stop:           []string{defaultStop},
		Temperature:    defaultTemperature,
		Rules:          []string{"dont use external libraries"}, // Example rule.
		UseLexer:       true,
	}

	// Pastel color codes (for terminal output).
	// Public access for CLI tool if needed, or keep private.
	ColorReset  = "\033[0m"
	ColorGreen  = "\033[38;5;119m" // Pastel green
	ColorYellow = "\033[38;5;220m" // Pastel yellow
	ColorBlue   = "\033[38;5;153m" // Pastel blue
	ColorRed    = "\033[38;5;203m" // Pastel red
	ColorCyan   = "\033[38;5;141m"
)

// =============================================================================
// Helper Functions
// =============================================================================

// prettyPrint prints a string in a pastel color.
func prettyPrint(color, text string) {
	fmt.Print(color, text, ColorReset)
}

// generatePrompt generates the prompt for the LLM.
func generatePrompt(codeSnippet string, config Config) string {
	// Truncate the code snippet if it exceeds the maximum content length.
	if len(codeSnippet) > maxContentLength {
		// Consider truncating intelligently (e.g., preserve function signature)
		codeSnippet = "..." + codeSnippet[len(codeSnippet)-maxContentLength+3:]
		log.Printf("Truncated code snippet to approximately %d characters.", maxContentLength)
	}
	return fmt.Sprintf(config.PromptTemplate, codeSnippet)
}

// OllamaError is a custom error type for Ollama-related errors.
type OllamaError struct {
	Message string
	Status  int // HTTP status code.
}

func (e *OllamaError) Error() string {
	return fmt.Sprintf("Ollama error: %s (Status: %d)", e.Message, e.Status)
}

// callOllamaAPI calls the Ollama API using net/http.
// Consider replacing with ollama-go library for simplification.
func callOllamaAPI(ctx context.Context, prompt string, config Config) (io.ReadCloser, error) {
	// --- Start of section potentially replaceable by ollama-go ---
	u, err := url.Parse(config.OllamaURL + "/api/generate")
	if err != nil {
		return nil, fmt.Errorf("error parsing Ollama URL: %w", err)
	}

	// Prepare the request payload.
	payload := map[string]interface{}{
		"model":  config.Model,
		"prompt": prompt,
		"stream": true, // Use streaming.
		"options": map[string]interface{}{ // Add options
			"temperature": config.Temperature,
			"num_ctx":     2048, // Context window size
			"top_p":       0.9,  // Nucleus sampling
			// Add other Ollama options as needed
		},
		"format": "", // Can be "json" if needed
		// "max_tokens": config.MaxTokens, // Note: Ollama generate API might not directly support max_tokens here, it's often in options or handled by model implicitly. Check Ollama docs.
		"stop": config.Stop,
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
	req.Header.Set("Accept", "application/x-ndjson") // Indicate expecting newline-delimited JSON

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		// Handle potential network errors (e.g., connection refused)
		return nil, fmt.Errorf("error making HTTP request to Ollama: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close() // Ensure body is closed on error
		bodyBytes, readErr := io.ReadAll(resp.Body)
		bodyString := ""
		if readErr == nil {
			bodyString = string(bodyBytes)
		} else {
			bodyString = fmt.Sprintf("(could not read error body: %v)", readErr)
		}
		// Check for specific Ollama errors if possible
		if resp.StatusCode == http.StatusNotFound {
			return nil, &OllamaError{
				Message: fmt.Sprintf("Ollama API endpoint not found or model '%s' not available: %s", config.Model, bodyString),
				Status:  resp.StatusCode,
			}
		}
		return nil, &OllamaError{
			Message: fmt.Sprintf("Ollama API request failed: %s", bodyString),
			Status:  resp.StatusCode,
		}
	}
	// --- End of section potentially replaceable by ollama-go ---
	return resp.Body, nil

	/*
	   // Example using ollama-go (replace above block)
	   client, err := ollamago.ClientFromEnvironment() // Or specify URL
	   if err != nil {
	       return nil, fmt.Errorf("failed to create ollama client: %w", err)
	   }

	   req := &ollamago.GenerateRequest{
	       Model:  config.Model,
	       Prompt: prompt,
	       Stream: ollamago.Ptr(true), // Use helper for pointer
	       Options: map[string]interface{}{
	           "temperature": config.Temperature,
	           "num_ctx":     2048,
	           "top_p":       0.9,
	           "stop":        config.Stop,
	       },
	   }

	   // Need a way to get ReadCloser from streaming response
	   // The library's streaming function takes a callback.
	   // We might need to adapt this part or use a pipe.
	   // For now, sticking with manual HTTP for direct ReadCloser.
	   // See ollama-go documentation for streaming patterns.
	*/
}

// streamCompletion reads the completion from the provided reader and streams
// it to the provided writer. It handles the Ollama newline-delimited JSON stream format.
// Consider replacing with ollama-go library's streaming handler.
func streamCompletion(ctx context.Context, r io.ReadCloser, w io.Writer) error {
	defer r.Close()
	reader := bufio.NewReader(r)

	for {
		// Check context cancellation frequently
		select {
		case <-ctx.Done():
			log.Println("Context cancelled during streaming")
			return ctx.Err()
		default:
			// Proceed with reading
		}

		line, err := reader.ReadBytes('\n') // Read until newline
		if err != nil {
			if err == io.EOF {
				// Check if there's any remaining data before EOF
				if len(line) > 0 {
					if processLine(line, w) != nil {
						// Log error processing final line, but EOF is primary
						log.Printf("Error processing final line before EOF: %v", err)
					}
				}
				return nil // End of stream reached successfully
			}
			// Log other read errors
			log.Printf("Error reading from Ollama stream: %v", err)
			return fmt.Errorf("error reading from Ollama stream: %w", err)
		}

		if err := processLine(line, w); err != nil {
			// Log error processing line and continue if possible, or return fatal error
			log.Printf("Error processing Ollama response line: %v", err)
			// Decide whether to continue or fail based on error type
			// For now, we continue, skipping the bad line.
			// return err // Uncomment to make processing errors fatal
		}
	}
}

// processLine unmarshals and handles a single line from the Ollama stream.
func processLine(line []byte, w io.Writer) error {
	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return nil // Skip empty lines
	}

	var resp OllamaResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		// Log the problematic line for debugging
		log.Printf("Error unmarshalling Ollama JSON line: %v, line content: '%s'", err, string(line))
		// Don't return error here, just skip the line to be robust against malformed JSON chunks
		return nil // Or return fmt.Errorf("error unmarshalling JSON: %w", err) to fail hard
	}

	// Check for errors reported within the JSON payload
	if resp.Error != "" {
		log.Printf("Ollama reported an error in stream: %s", resp.Error)
		return fmt.Errorf("ollama stream error: %s", resp.Error)
	}

	// Write the actual response content to the output writer
	if _, err := fmt.Fprint(w, resp.Response); err != nil {
		log.Printf("Error writing completion chunk to output: %v", err)
		return fmt.Errorf("error writing to output: %w", err)
	}

	// Check if Ollama signaled completion
	if resp.Done {
		// This might indicate the end, but rely on EOF from streamCompletion's ReadBytes
		// log.Println("Ollama signalled done in stream.")
	}

	return nil
}

// =============================================================================
// Main Functions
// =============================================================================

// GetCompletion returns the completion from Ollama for the given code snippet (non-streaming).
// This function buffers the entire response. Use streaming functions for large completions.
func GetCompletion(ctx context.Context, codeSnippet string, config Config) (string, error) {
	prompt := generatePrompt(codeSnippet, config)

	var ollamaResponse string
	err := retry(ctx, func() error {
		reader, err := callOllamaAPI(ctx, prompt, config)
		if err != nil {
			// Check if the error is retryable (e.g., 503)
			var ollamaErr *OllamaError
			if errors.As(err, &ollamaErr) && (ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests) {
				return err // Signal retry for specific HTTP errors
			}
			// Use errors.Join for Go 1.20+ or wrap otherwise
			return fmt.Errorf("non-retryable error from callOllamaAPI: %w", err) // Wrap for clarity and prevent retry
		}

		// Buffer the streaming response
		var buffer bytes.Buffer
		if err := streamCompletion(ctx, reader, &buffer); err != nil {
			// Check if the stream error is retryable (might be network related)
			// For now, assume stream errors are not retryable unless specifically identified
			return fmt.Errorf("error during stream processing: %w", err) // Wrap, non-retryable
		}
		ollamaResponse = buffer.String()
		return nil // Success
	}, maxRetries, retryDelay)

	if err != nil {
		return "", fmt.Errorf("failed to get completion after retries: %w", err) // Wrap final error
	}

	return ollamaResponse, nil
}

// retry is a helper function for retrying operations with exponential backoff (optional).
func retry(ctx context.Context, operation func() error, maxRetries int, initialDelay time.Duration) error {
	var err error
	currentDelay := initialDelay
	for i := 0; i < maxRetries; i++ {
		err = operation()
		if err == nil {
			return nil // Success
		}

		// Check if the error suggests a retry is warranted.
		// Example: Check for specific error types or messages.
		var ollamaErr *OllamaError
		isRetryable := errors.As(err, &ollamaErr) && (ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests)
		// Add other retryable conditions if needed (e.g., temporary network errors)

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
			// Exponential backoff (optional)
			// currentDelay *= 2
			// Add jitter: currentDelay += time.Duration(rand.Intn(int(initialDelay)))
		}
	}
	// If loop completes, all retries failed
	log.Printf("Operation failed after %d retries.", maxRetries)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, err) // Return the last error
}

// ProcessCodeWithMiddleware applies middleware functions sequentially.
func ProcessCodeWithMiddleware(codeSnippet string, middleware ...Middleware) (string, error) {
	currentCode := codeSnippet
	for i, m := range middleware {
		output, err := m(currentCode)
		if err != nil {
			// Provide context about which middleware failed
			return "", fmt.Errorf("middleware %d failed: %w", i+1, err)
		}
		currentCode = output
	}
	return currentCode, nil
}

// LexGoCode uses go/scanner to lex Go code in-process.
func LexGoCode(code string) (LexerResult, error) {
	fset := token.NewFileSet() // Positions are relative to fset
	// Add a dummy file to the fileset to associate positions with a filename (optional but good practice)
	file := fset.AddFile("input.go", fset.Base(), len(code))

	var s scanner.Scanner
	// Initialize the scanner with the file, source code, error handler, and mode.
	// The error handler can log errors or collect them.
	var scanErrs []error
	errHandler := func(pos token.Position, msg string) {
		scanErrs = append(scanErrs, fmt.Errorf("scanner error at %s: %s", pos.String(), msg))
	}
	// scanner.ScanComments includes comments as tokens
	s.Init(file, []byte(code), errHandler, scanner.ScanComments)

	result := LexerResult{
		Tokens:    make([]string, 0),
		Positions: make([]token.Pos, 0),
		Kinds:     make([]token.Token, 0),
	}

	// Scan the code token by token
	for {
		pos, tok, lit := s.Scan()
		if tok == token.EOF {
			break // End of file
		}
		// Record the token information
		result.Positions = append(result.Positions, pos)
		result.Kinds = append(result.Kinds, tok)
		// lit holds the literal value (e.g., "myVar", "123", "\"hello\"")
		// For keywords, operators, etc., lit might be empty, use tok.String()
		tokenText := lit
		if tokenText == "" {
			tokenText = tok.String()
		}
		result.Tokens = append(result.Tokens, tokenText)
	}

	// Check if any scanning errors occurred
	if len(scanErrs) > 0 {
		// Combine multiple errors if necessary
		// For simplicity, return the first error encountered.
		return result, fmt.Errorf("lexing failed: %w", scanErrs[0]) // Or combine errors
	}

	return result, nil
}

// GetCompletionWithLexer uses the go/scanner lexer to get context before calling LLM.
// This implementation needs refinement on how to best use token context.
func GetCompletionWithLexer(ctx context.Context, codeSnippet string, config Config) (string, error) {
	lexerResult, err := LexGoCode(codeSnippet)
	if err != nil {
		// Log the lexing error but potentially fall back to basic completion
		log.Printf("Lexing failed, falling back to basic completion: %v", err)
		// Fallback: Use the raw code snippet if lexing fails
		return GetCompletion(ctx, codeSnippet, config)
		// Or return the error:
		// return "", fmt.Errorf("failed to lex code for context: %w", err)
	}

	// --- Strategy for using lexer context ---
	// This is a simple example: use the last N tokens.
	// More sophisticated strategies could involve finding the current scope,
	// function signature, variable types, etc., using go/parser.
	contextTokenCount := 15 // Number of recent tokens to consider
	startIndex := 0
	if len(lexerResult.Tokens) > contextTokenCount {
		startIndex = len(lexerResult.Tokens) - contextTokenCount
	}
	contextTokens := lexerResult.Tokens[startIndex:]

	// Reconstruct code snippet from relevant tokens.
	// This might need adjustment based on token kinds (e.g., spacing)
	contextCode := strings.Join(contextTokens, " ") // Simple join with space

	log.Printf("Using lexer context: [%s]", contextCode) // Log context used

	// Generate prompt using the lexer-derived context
	prompt := generatePrompt(contextCode, config)

	// Call Ollama with the context-aware prompt (non-streaming version)
	// This part is similar to GetCompletion but uses the lexer-derived prompt.
	var ollamaResponse string
	err = retry(ctx, func() error {
		reader, err := callOllamaAPI(ctx, prompt, config)
		if err != nil {
			var ollamaErr *OllamaError
			if errors.As(err, &ollamaErr) && (ollamaErr.Status == http.StatusServiceUnavailable || ollamaErr.Status == http.StatusTooManyRequests) {
				return err // Retryable
			}
			return fmt.Errorf("non-retryable error from callOllamaAPI: %w", err)
		}

		var buffer bytes.Buffer
		if err := streamCompletion(ctx, reader, &buffer); err != nil {
			return fmt.Errorf("error during stream processing: %w", err)
		}
		ollamaResponse = buffer.String()
		return nil
	}, maxRetries, retryDelay)

	if err != nil {
		return "", fmt.Errorf("failed to get completion with lexer context: %w", err)
	}
	return ollamaResponse, nil
}

// --- Middleware Examples ---

// lintResult checks for balanced braces (simple example).
func lintResult(input string) (string, error) {
	openBraces := strings.Count(input, "{")
	closeBraces := strings.Count(input, "}")
	if openBraces > closeBraces {
		// LLM might have stopped mid-block, potentially okay depending on use case
		log.Printf("Lint warning: potentially unbalanced braces (open > close)")
	} else if closeBraces > openBraces {
		log.Printf("Lint warning: potentially unbalanced braces (close > open)")
		// return "", fmt.Errorf("lint error: unmatched closing braces '}'") // Make it an error if desired
	}
	// Add more sophisticated linting if needed (e.g., using go/parser)
	return input, nil
}

// removeExternalLibraries attempts to filter import lines (basic example).
func removeExternalLibraries(input string) (string, error) {
	// This is very basic and might remove valid code.
	// A proper implementation would parse the import block.
	lines := strings.Split(input, "\n")
	filteredLines := make([]string, 0, len(lines))
	inImportBlock := false
	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "import (") {
			inImportBlock = true
			continue // Skip the start of the block
		}
		if inImportBlock && strings.HasPrefix(trimmedLine, ")") {
			inImportBlock = false
			continue // Skip the end of the block
		}
		if inImportBlock {
			// Check if the import path looks like an external library
			if strings.Contains(trimmedLine, `"`) && !strings.Contains(trimmedLine, `"fmt"`) && !strings.Contains(trimmedLine, `"os"`) /* add more stdlib */ {
				log.Printf("Filtering potential external library import: %s", trimmedLine)
				continue // Skip this line
			}
		}
		// Simple check for single-line imports
		if strings.HasPrefix(trimmedLine, `import "`) && !strings.Contains(trimmedLine, `"fmt"`) && !strings.Contains(trimmedLine, `"os"`) /* add more stdlib */ {
			log.Printf("Filtering potential external library import: %s", trimmedLine)
			continue // Skip this line
		}
		filteredLines = append(filteredLines, line)
	}
	return strings.Join(filteredLines, "\n"), nil
}

// --- Spinner ---

// Spinner struct to manage the spinner animation.
type Spinner struct {
	chars    []string
	index    int
	mu       sync.Mutex
	stopChan chan struct{}
	running  bool // Track if the spinner goroutine is active
}

// NewSpinner creates a new Spinner.
func NewSpinner() *Spinner {
	return &Spinner{
		chars:    []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}, // Extended chars
		index:    0,
		stopChan: make(chan struct{}), // Initialize channel
	}
}

// Start starts the spinner animation in a goroutine.
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

// Stop stops the spinner animation.
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

// --- File Handling and Streaming Completion ---

// GetCompletionFromFile reads code up to a point and gets completion (non-streaming version).
// Consider using GetCompletionStreamFromFile for interactive use.
func GetCompletionFromFile(ctx context.Context, filename string, row, col int, config Config) (string, error) {
	codeSnippet, err := extractCodeSnippet(filename, row, col)
	if err != nil {
		return "", err // Error includes context (file reading, position invalid)
	}

	// Decide whether to use lexer based on config
	if config.UseLexer {
		// GetCompletionWithLexer currently returns the full completion string
		return GetCompletionWithLexer(ctx, codeSnippet, config)
	}
	// GetCompletion returns the full completion string
	return GetCompletion(ctx, codeSnippet, config)
}

// GetCompletionStreamFromFile reads code, extracts comments, and streams completion.
func GetCompletionStreamFromFile(ctx context.Context, filename string, row, col int, config Config, w io.Writer) error {
	code, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("error reading file '%s': %w", filename, err)
	}
	codeStr := string(code)

	// Extract the relevant part of the code for the prompt
	codeSnippet, err := extractCodeSnippet(filename, row, col) // Reuses the extraction logic
	if err != nil {
		return err // Error includes context
	}

	// Read comments (using basic approach for now)
	comments := readComments(codeStr) // Pass the full code content
	if len(comments) > 0 {
		// Use prettyPrint or direct Fprintf with colors
		fmt.Fprintf(w, "%sComments:%s\n", ColorYellow, ColorReset)
		for _, comment := range comments {
			fmt.Fprintf(w, "%s  %s%s\n", ColorBlue, comment, ColorReset)
		}
		fmt.Fprintln(w) // Add a newline for separation
	}

	// --- Prepare and call Ollama for streaming ---
	prompt := generatePrompt(codeSnippet, config)

	// Get the streaming reader
	reader, err := callOllamaAPI(ctx, prompt, config)
	if err != nil {
		// Attempt to provide more context about the error
		var ollamaErr *OllamaError
		if errors.As(err, &ollamaErr) {
			return fmt.Errorf("failed to initiate Ollama stream (Status: %d): %w", ollamaErr.Status, err)
		}
		return fmt.Errorf("failed to initiate Ollama stream: %w", err)
	}

	// Stream the completion directly to the provided writer (e.g., os.Stdout)
	fmt.Fprintf(w, "%sCompletion:%s\n", ColorGreen, ColorReset) // Indicate start of completion
	err = streamCompletion(ctx, reader, w)
	fmt.Fprintln(w) // Add a final newline after streaming finishes
	if err != nil {
		// Check for context cancellation vs other stream errors
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return fmt.Errorf("completion stream cancelled or timed out: %w", err)
		}
		return fmt.Errorf("error during completion streaming: %w", err)
	}

	return nil // Success
}

// extractCodeSnippet reads a file and extracts code up to a given row/col.
func extractCodeSnippet(filename string, row, col int) (string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return "", fmt.Errorf("error opening file '%s': %w", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var lines []string
	currentLine := 1
	var targetLine string
	foundLine := false

	// Read lines up to the target row
	for scanner.Scan() {
		lineText := scanner.Text()
		if currentLine == row {
			targetLine = lineText
			foundLine = true
			// Keep reading lines for potential multi-line context later? No, prompt uses only up to cursor.
			// lines = append(lines, lineText) // Keep line for context if needed
		} else if currentLine < row {
			// Store previous lines if needed for broader context (not used in current prompt)
			// lines = append(lines, lineText)
		}
		currentLine++
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error reading file '%s': %w", filename, err)
	}

	if !foundLine {
		return "", fmt.Errorf("row %d not found in file '%s' (file has %d lines)", row, filename, currentLine-1)
	}

	// Validate column (using 1-based indexing for user input)
	// Convert col to 0-based index for slicing. Be careful with multi-byte chars.
	// For simplicity, assuming byte-based column index for now.
	if col <= 0 || col > len(targetLine)+1 { // Allow col to be just after the last char
		return "", fmt.Errorf("column %d is invalid for row %d in file '%s' (line length: %d)", col, row, filename, len(targetLine))
	}

	// Extract the snippet from the target line up to the column (0-based index)
	codeSnippet := targetLine[:col-1]

	// --- Optional: Add context from previous lines ---
	// prefixContext := ""
	// contextLines := 5 // Number of previous lines to include
	// startLine := len(lines) - contextLines
	// if startLine < 0 {
	//     startLine = 0
	// }
	// if len(lines) > 0 {
	//     prefixContext = strings.Join(lines[startLine:], "\n") + "\n"
	// }
	// return prefixContext + codeSnippet, nil
	// --- End Optional Context ---

	return codeSnippet, nil
}

// readComments reads comments from the code (basic implementation).
// TODO: Improve using go/scanner or go/parser for accuracy.
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
