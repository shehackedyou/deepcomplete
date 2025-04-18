package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	// NOTE: Replace with your actual module path
	"github.com/shehackedyou/deepcomplete"
)

// Simple version string (update as needed).
const version = "0.0.1" // Added in Cycle 4

func main() {
	// --- Configuration Loading ---
	// Load config, using defaults as fallback and attempting to write default if none found.
	config, configLoadErr := deepcomplete.LoadConfig()
	if configLoadErr != nil && !errors.Is(configLoadErr, deepcomplete.ErrConfig) {
		// Log fatal errors (e.g., cannot determine paths).
		log.Fatalf("Fatal error loading initial config: %v", configLoadErr)
	} else if configLoadErr != nil {
		// Log non-fatal warnings (e.g., failed write, parse errors) but continue.
		log.Printf("%sWarning during initial config load: %v. Using defaults and flags.%s", deepcomplete.ColorYellow, configLoadErr, deepcomplete.ColorReset)
		// Ensure config is reset to default if load partially failed but wasn't fatal.
		if err := config.Validate(); err != nil {
			config = deepcomplete.DefaultConfig // Fallback to pure defaults.
		}
	}

	// --- Flag Definition ---
	// Define flags using loaded config values as defaults.
	ollamaURL := flag.String("ollama-url", config.OllamaURL, "URL of the Ollama API endpoint.")
	model := flag.String("model", config.Model, "Name of the Ollama model to use.")
	row := flag.Int("row", 0, "Cursor row number (1-based). Required with --file.")
	col := flag.Int("col", 0, "Cursor column number (1-based, byte offset). Required with --file.")
	filename := flag.String("file", "", "Path to Go file for context-aware completion.")
	useAst := flag.Bool("ast", config.UseAst, "Enable AST/type analysis for richer context.")
	useFim := flag.Bool("fim", config.UseFim, "Use Fill-in-the-Middle prompting.")
	maxTokens := flag.Int("max-tokens", config.MaxTokens, "Max tokens for completion response.")
	temperature := flag.Float64("temperature", config.Temperature, "Sampling temperature for Ollama.")
	stop := flag.String("stop", strings.Join(config.Stop, ","), "Comma-separated stop sequences.")
	maxPreambleLen := flag.Int("max-preamble-len", config.MaxPreambleLen, "Max bytes for context preamble.")
	maxSnippetLen := flag.Int("max-snippet-len", config.MaxSnippetLen, "Max bytes for code snippet context.")
	versionFlag := flag.Bool("version", false, "Print version information and exit.") // Added Cycle 4

	flag.Parse() // Parse command-line flags.

	// --- Handle Version Flag (Added Cycle 4) ---
	if *versionFlag {
		fmt.Printf("deepcomplete-cli version %s\n", version)
		os.Exit(0) // Exit immediately after printing version.
	}

	// --- Apply Parsed Flag Values to Config ---
	// Update config struct with final values (Flag > Config File > Default).
	config.OllamaURL = *ollamaURL
	config.Model = *model
	config.UseAst = *useAst
	config.UseFim = *useFim
	config.MaxTokens = *maxTokens
	config.Temperature = *temperature
	config.MaxPreambleLen = *maxPreambleLen
	config.MaxSnippetLen = *maxSnippetLen

	// Reprocess stop sequences based on final flag value.
	stopSequences := strings.Split(*stop, ",")
	validStopSequences := make([]string, 0, len(stopSequences))
	for _, s := range stopSequences {
		trimmed := strings.TrimSpace(s)
		if trimmed != "" {
			validStopSequences = append(validStopSequences, trimmed)
		}
	}
	// Override config only if flag provided non-empty sequences OR was explicitly passed.
	if len(validStopSequences) > 0 || isFlagPassed("stop") {
		config.Stop = validStopSequences
	}

	// --- Final Config Validation ---
	if err := config.Validate(); err != nil {
		log.Fatalf("%sError in final configuration: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
	}
	log.Printf("Final config: %+v", config) // Log effective config.

	// --- Determine Input Mode (File vs Prompt Args) ---
	var promptArgs string
	if *filename == "" && flag.NArg() > 0 {
		promptArgs = strings.Join(flag.Args(), " ")
		log.Printf("Using non-flag arguments as prompt: %q", promptArgs)
	}

	// Create background context.
	ctx := context.Background()

	// Create spinner for user feedback.
	spinner := deepcomplete.NewSpinner()

	// --- Input Validation ---
	if *filename != "" && (*row <= 0 || *col <= 0) {
		log.Fatalf("%sError: Both --row and --col must be positive integers when --file is provided.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
	if *filename == "" && promptArgs == "" {
		log.Fatalf("%sError: Either provide a prompt as arguments or use --file (with --row and --col).%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
	if *filename != "" { // Check if file exists if provided.
		if _, err := os.Stat(*filename); errors.Is(err, os.ErrNotExist) {
			log.Fatalf("%sError: File not found: %s%s", deepcomplete.ColorRed, *filename, deepcomplete.ColorReset)
		} else if err != nil {
			log.Fatalf("%sError checking file '%s': %v%s", deepcomplete.ColorRed, *filename, err, deepcomplete.ColorReset)
		}
	}

	// --- Initialize DeepCompleter Service ---
	completer, err := deepcomplete.NewDeepCompleterWithConfig(config)
	if err != nil {
		log.Fatalf("%sError initializing completer: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
	}
	defer func() { // Ensure resources are closed on exit.
		log.Println("Closing completer resources...")
		if closeErr := completer.Close(); closeErr != nil {
			log.Printf("%sError closing completer: %v%s", deepcomplete.ColorRed, closeErr, deepcomplete.ColorReset)
		}
	}()

	// --- Execute Completion Logic ---
	var completionErr error

	switch {
	case promptArgs != "": // Direct prompt input from args.
		if config.UseFim {
			log.Fatalf("%sError: Fill-in-the-Middle (-fim) requires file input (--file).%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
		}
		spinner.Start("Waiting for Ollama...")
		var completion string
		completion, completionErr = completer.GetCompletion(ctx, promptArgs)
		spinner.Stop()
		if completionErr != nil {
			log.Fatalf("%sError getting completion: %v%s", deepcomplete.ColorRed, completionErr, deepcomplete.ColorReset)
		}
		deepcomplete.PrettyPrint(deepcomplete.ColorGreen, "Completion:\n")
		fmt.Println(completion) // Print raw completion to stdout.

	case *filename != "": // File input (uses streaming and potentially AST/FIM).
		initialSpinnerMsg := "Waiting for Ollama..."
		if config.UseAst {
			initialSpinnerMsg = "Analyzing context..."
		}
		spinner.Start(initialSpinnerMsg)

		// Update spinner message after delay if analysis is enabled.
		if config.UseAst {
			go func() {
				time.Sleep(750 * time.Millisecond)
				// Spinner update is best-effort, might update after stop.
				spinner.UpdateMessage("Waiting for Ollama...")
			}()
		}

		// Call streaming completion, writing directly to os.Stdout.
		completionErr = completer.GetCompletionStreamFromFile(ctx, *filename, *row, *col, os.Stdout)
		spinner.Stop() // Stop spinner after streaming finishes or errors.
		if completionErr != nil {
			// Error message already includes color codes if coming from deepcomplete helpers.
			log.Fatalf("Error getting completion from file: %v", completionErr)
		}
		// Output is already streamed, exit normally on success.
		return

	default:
		// Should be unreachable due to validation above.
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
