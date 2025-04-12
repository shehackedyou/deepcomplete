// cmd/deepcomplete-cli/main.go
package main

import (
	"context"
	"errors" // For os.IsNotExist
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	// Use the CORRECT module path provided by the user
	"github.com/shehackedyou/deepcomplete"
)

func main() {
	// Define command-line flags with better help messages.
	// Use exported DefaultConfig values for defaults
	ollamaURL := flag.String("ollama-url", deepcomplete.DefaultConfig.OllamaURL, "URL of the Ollama API endpoint.")
	model := flag.String("model", deepcomplete.DefaultConfig.Model, "Name of the Ollama model to use for completion.")
	row := flag.Int("row", 0, "Cursor row number in the file (1-based). Required with --file.")
	col := flag.Int("col", 0, "Cursor column number in the file (1-based, byte offset). Required with --file.")
	filename := flag.String("file", "", "Path to the Go file for context-aware completion.")
	prompt := flag.String("prompt", "", "A simple code prompt for basic completion (overrides --file).")
	useAst := flag.Bool("ast", deepcomplete.DefaultConfig.UseAst, "Enable Abstract Syntax Tree (AST) and type analysis for richer context.")
	// lexer flag removed
	maxTokens := flag.Int("max-tokens", deepcomplete.DefaultConfig.MaxTokens, "Maximum number of tokens for the completion response.")
	temperature := flag.Float64("temperature", deepcomplete.DefaultConfig.Temperature, "Sampling temperature for the Ollama model (e.g., 0.1 for more deterministic, 0.8 for more creative).")
	stop := flag.String("stop", strings.Join(deepcomplete.DefaultConfig.Stop, ","), "Comma-separated sequences where the model should stop generating text.")
	// Add a simple version flag? (Example - not fully implemented)
	// version := flag.Bool("version", false, "Print version information and exit.")

	flag.Parse()

	// Handle version flag if implemented
	// if *version {
	// 	fmt.Println("deepcomplete-cli version 0.1.0 (example)")
	// 	os.Exit(0)
	// }

	// --- Configuration Setup ---
	// Start with default config
	config := deepcomplete.DefaultConfig
	// Override specific fields from flags
	config.OllamaURL = *ollamaURL
	config.Model = *model
	config.UseAst = *useAst
	// config.UseLexer removed
	config.MaxTokens = *maxTokens
	config.Temperature = *temperature
	// Ensure stop sequences are trimmed of whitespace
	stopSequences := strings.Split(*stop, ",")
	for i := range stopSequences {
		stopSequences[i] = strings.TrimSpace(stopSequences[i])
	}
	// Filter out empty strings that might result from trailing commas
	validStopSequences := make([]string, 0, len(stopSequences))
	for _, s := range stopSequences {
		if s != "" {
			validStopSequences = append(validStopSequences, s)
		}
	}
	config.Stop = validStopSequences

	if !config.UseAst {
		log.Println("AST analysis disabled via flag.")
	}

	// Create a background context. Consider adding timeout later.
	ctx := context.Background()

	// Create a spinner.
	spinner := deepcomplete.NewSpinner()

	// --- Middleware Setup ---
	// Middleware can be added here if needed later.
	// middleware := []deepcomplete.Middleware{}

	// --- Input Validation ---
	if *filename != "" && (*row <= 0 || *col <= 0) {
		// Use exported colors for error message
		log.Fatalf("%sError: Both --row and --col must be positive integers when --file is provided.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
	if *filename == "" && *prompt == "" {
		log.Fatalf("%sError: Either --prompt or --file (with --row and --col) must be provided.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
	if *filename != "" && *prompt != "" {
		log.Printf("%sWarning: Both --prompt and --file provided. Using --file.%s", deepcomplete.ColorYellow, deepcomplete.ColorReset)
		*prompt = "" // Prioritize file input
	}
	// Check if file exists if provided
	if *filename != "" {
		if _, err := os.Stat(*filename); errors.Is(err, os.ErrNotExist) {
			log.Fatalf("%sError: File not found: %s%s", deepcomplete.ColorRed, *filename, deepcomplete.ColorReset)
		} else if err != nil {
			log.Fatalf("%sError checking file '%s': %v%s", deepcomplete.ColorRed, *filename, err, deepcomplete.ColorReset)
		}
	}

	// --- Execute Completion Logic ---
	var err error

	switch {
	case *prompt != "": // Direct prompt input (uses basic context)
		spinner.Start("Waiting for Ollama...") // Updated spinner message
		var completion string
		completion, err = deepcomplete.GetCompletion(ctx, *prompt, config) // Use simplified basic completion
		spinner.Stop()
		if err != nil {
			// Use exported colors
			log.Fatalf("%sError getting completion: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
		}
		// Middleware processing removed from default path for simplicity
		// processedCompletion, mwErr := deepcomplete.ProcessCodeWithMiddleware(completion, middleware...)
		// if mwErr != nil {
		// 	log.Fatalf("%sError processing completion: %v%s", deepcomplete.ColorRed, mwErr, deepcomplete.ColorReset)
		// }

		// Use exported PrettyPrint helper
		deepcomplete.PrettyPrint(deepcomplete.ColorGreen, "Completion:\n")
		fmt.Println(completion) // Print raw completion

	case *filename != "": // File input (uses streaming and potentially AST)
		spinner.Start("Analyzing context...") // Initial spinner message
		// Use the CORRECT exported streaming function name
		// Pass a callback to update spinner message? Or just let it run.
		// For simplicity, keep the single message for now.
		// TODO: Update spinner message before calling Ollama if analysis takes noticeable time
		err = deepcomplete.GetCompletionStreamFromFile(ctx, *filename, *row, *col, config, os.Stdout)
		spinner.Stop() // Stop spinner after streaming finishes or errors
		if err != nil {
			// Use exported colors
			log.Fatalf("%sError getting completion from file: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
		}
		// Output is already streamed, so just exit normally on success
		return

	default:
		// This case should be unreachable due to validation above, but keep as safeguard
		log.Fatalf("%sInternal Error: Invalid flag combination.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
}
