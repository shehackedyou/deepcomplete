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
	"time" // For potential delays if needed

	// Use your actual module path here
	"github.com/shehackedyou/deepcomplete" // Corrected import path
)

func main() {
	// --- Configuration Loading ---
	// Load config from file first, using defaults as fallback
	// This now also attempts to write the default config if none is found.
	config, configLoadErr := deepcomplete.LoadConfig()
	if configLoadErr != nil && !errors.Is(configLoadErr, deepcomplete.ErrConfig) {
		// Only log fatal errors here (e.g., cannot determine paths)
		// Non-fatal ErrConfig (like failed write/parse) allows proceeding with defaults.
		log.Fatalf("Fatal error loading initial config: %v", configLoadErr)
	} else if configLoadErr != nil {
		// Log non-fatal warnings (e.g., failed write, parse errors) but continue
		log.Printf("%sWarning during initial config load: %v. Using defaults and flags.%s", deepcomplete.ColorYellow, configLoadErr, deepcomplete.ColorReset)
		// Ensure config is reset to default if load partially failed but wasn't fatal
		if err := config.Validate(); err != nil {
			config = deepcomplete.DefaultConfig // Fallback to pure defaults
		}
	}

	// --- Flag Definition ---
	// Define flags, using the loaded config values as the *default* values for the flags.
	ollamaURL := flag.String("ollama-url", config.OllamaURL, "URL of the Ollama API endpoint.")
	model := flag.String("model", config.Model, "Name of the Ollama model to use for completion.")
	row := flag.Int("row", 0, "Cursor row number in the file (1-based). Required with --file.")
	col := flag.Int("col", 0, "Cursor column number in the file (1-based, byte offset). Required with --file.")
	filename := flag.String("file", "", "Path to the Go file for context-aware completion.")
	useAst := flag.Bool("ast", config.UseAst, "Enable Abstract Syntax Tree (AST) and type analysis for richer context.")
	useFim := flag.Bool("fim", config.UseFim, "Use Fill-in-the-Middle prompting instead of standard completion.")
	maxTokens := flag.Int("max-tokens", config.MaxTokens, "Maximum number of tokens for the completion response.")
	temperature := flag.Float64("temperature", config.Temperature, "Sampling temperature for the Ollama model.")
	stop := flag.String("stop", strings.Join(config.Stop, ","), "Comma-separated sequences where the model should stop generating text.")
	// Flags for new config options (Step 8)
	maxPreambleLen := flag.Int("max-preamble-len", config.MaxPreambleLen, "Maximum number of bytes for the context preamble.")
	maxSnippetLen := flag.Int("max-snippet-len", config.MaxSnippetLen, "Maximum number of bytes for the code snippet context.")
	// version := flag.Bool("version", false, "Print version information and exit.")

	flag.Parse() // Parse the flags

	// --- Apply Parsed Flag Values to Config ---
	// Update the config struct with final values (Flag > Config File > Default).
	config.OllamaURL = *ollamaURL
	config.Model = *model
	config.UseAst = *useAst
	config.UseFim = *useFim
	config.MaxTokens = *maxTokens
	config.Temperature = *temperature
	config.MaxPreambleLen = *maxPreambleLen // Apply new flag (Step 8)
	config.MaxSnippetLen = *maxSnippetLen   // Apply new flag (Step 8)

	// Reprocess stop sequences based on the final flag value
	stopSequences := strings.Split(*stop, ",")
	validStopSequences := make([]string, 0, len(stopSequences))
	for _, s := range stopSequences {
		trimmed := strings.TrimSpace(s)
		if trimmed != "" {
			validStopSequences = append(validStopSequences, trimmed)
		}
	}
	// Override only if flag provided non-empty sequences OR was explicitly passed
	if len(validStopSequences) > 0 || isFlagPassed("stop") {
		config.Stop = validStopSequences
	}

	// --- Final Config Validation ---
	if err := config.Validate(); err != nil {
		log.Fatalf("%sError in final configuration: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
	}

	log.Printf("Final config: %+v", config) // Log final effective config

	// --- Determine Input Mode (File vs Prompt Args) ---
	var promptArgs string
	if *filename == "" && flag.NArg() > 0 {
		promptArgs = strings.Join(flag.Args(), " ")
		log.Printf("Using non-flag arguments as prompt: %q", promptArgs)
	}

	// Create a background context. Consider adding timeout later.
	ctx := context.Background() // Keep main context simple for CLI

	// Create a spinner.
	spinner := deepcomplete.NewSpinner()

	// --- Input Validation ---
	if *filename != "" && (*row <= 0 || *col <= 0) {
		log.Fatalf("%sError: Both --row and --col must be positive integers when --file is provided.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
	if *filename == "" && promptArgs == "" {
		log.Fatalf("%sError: Either provide a prompt as arguments or use --file (with --row and --col).%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
	// Check if file exists if provided
	if *filename != "" {
		if _, err := os.Stat(*filename); errors.Is(err, os.ErrNotExist) {
			log.Fatalf("%sError: File not found: %s%s", deepcomplete.ColorRed, *filename, deepcomplete.ColorReset)
		} else if err != nil {
			log.Fatalf("%sError checking file '%s': %v%s", deepcomplete.ColorRed, *filename, err, deepcomplete.ColorReset)
		}
	}

	// --- Initialize DeepCompleter Service ---
	// Use the final, merged configuration
	// completer, err := deepcomplete.NewDeepCompleterWithConfig(config) // Old way
	completer, err := deepcomplete.NewDeepCompleterWithConfig(config) // Use constructor with merged config (Step 24 Change)
	if err != nil {
		log.Fatalf("%sError initializing completer: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
	}
	// --- Defer Close for Resource Cleanup (Step 24 Change) ---
	defer func() {
		log.Println("Closing completer resources...")
		if closeErr := completer.Close(); closeErr != nil {
			log.Printf("%sError closing completer: %v%s", deepcomplete.ColorRed, closeErr, deepcomplete.ColorReset)
		}
	}()
	// ---

	// --- Execute Completion Logic ---
	var completionErr error // Declare err here for use in the switch cases

	switch {
	case promptArgs != "": // Direct prompt input from args
		if config.UseFim {
			log.Fatalf("%sError: Fill-in-the-Middle (-fim) requires file input (--file), not a direct prompt.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
		}
		spinner.Start("Waiting for Ollama...")
		var completion string
		// Call the method on the completer instance
		completion, completionErr = completer.GetCompletion(ctx, promptArgs) // Assign to completionErr
		spinner.Stop()                                                       // Stop spinner *after* completion returns or errors
		if completionErr != nil {
			log.Fatalf("%sError getting completion: %v%s", deepcomplete.ColorRed, completionErr, deepcomplete.ColorReset)
		}

		// Use exported PrettyPrint helper
		deepcomplete.PrettyPrint(deepcomplete.ColorGreen, "Completion:\n")
		fmt.Println(completion) // Print raw completion

	case *filename != "": // File input (uses streaming and potentially AST/FIM)
		initialSpinnerMsg := "Waiting for Ollama..."
		if config.UseAst {
			initialSpinnerMsg = "Analyzing context..."
		}
		spinner.Start(initialSpinnerMsg)

		// Update spinner message after a short delay if analysis is enabled
		if config.UseAst {
			go func() {
				time.Sleep(750 * time.Millisecond)
				// Check if spinner is still running before updating
				// This requires making spinner.running accessible or adding an IsRunning method
				// For simplicity, assume it might update even if stopped just before this.
				spinner.UpdateMessage("Waiting for Ollama...")
			}()
		}

		// Call the method on the completer instance
		completionErr = completer.GetCompletionStreamFromFile(ctx, *filename, *row, *col, os.Stdout) // Assign to completionErr
		spinner.Stop()                                                                               // Stop spinner *after* streaming finishes or errors
		if completionErr != nil {
			// Error message already includes color codes if coming from deepcomplete helpers
			log.Fatalf("Error getting completion from file: %v", completionErr)
		}
		// Output is already streamed, so just exit normally on success
		return // Successful stream exit

	default:
		// This case should be unreachable due to validation above, but keep as safeguard
		log.Fatalf("%sInternal Error: Invalid input combination.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
}

// isFlagPassed checks if a flag was explicitly set on the command line.
func isFlagPassed(name string) bool {
	found := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
}
