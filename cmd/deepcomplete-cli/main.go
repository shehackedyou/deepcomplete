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
	if configLoadErr != nil {
		// Log error loading config file, but continue with defaults + flags
		log.Printf("%sWarning: Failed to load or write config file: %v. Using defaults and flags.%s", deepcomplete.ColorYellow, configLoadErr, deepcomplete.ColorReset)
	}

	// --- Flag Definition ---
	// Define flags, using the loaded config values as the *default* values for the flags.
	// This way, if a flag is not set on the command line, the value from the config file (or default) is used.
	// If a flag *is* set, its value will override the config file/default.
	ollamaURL := flag.String("ollama-url", config.OllamaURL, "URL of the Ollama API endpoint.")
	model := flag.String("model", config.Model, "Name of the Ollama model to use for completion.")
	row := flag.Int("row", 0, "Cursor row number in the file (1-based). Required with --file.")
	col := flag.Int("col", 0, "Cursor column number in the file (1-based, byte offset). Required with --file.")
	filename := flag.String("file", "", "Path to the Go file for context-aware completion.")
	// prompt flag removed
	useAst := flag.Bool("ast", config.UseAst, "Enable Abstract Syntax Tree (AST) and type analysis for richer context.")
	fim := flag.Bool("fim", config.UseFim, "Use Fill-in-the-Middle prompting instead of standard completion.")
	maxTokens := flag.Int("max-tokens", config.MaxTokens, "Maximum number of tokens for the completion response.")
	temperature := flag.Float64("temperature", config.Temperature, "Sampling temperature for the Ollama model (e.g., 0.1 for more deterministic, 0.8 for more creative).")
	stop := flag.String("stop", strings.Join(config.Stop, ","), "Comma-separated sequences where the model should stop generating text.")
	// version := flag.Bool("version", false, "Print version information and exit.")

	flag.Parse() // Parse the flags

	// --- Apply Parsed Flag Values to Config ---
	// After parsing, the flag variables (*ollamaURL, *model, etc.) hold the final value
	// based on the precedence: command-line flag > config file value > default value.
	// Update the config struct with these final values.
	config.OllamaURL = *ollamaURL
	config.Model = *model
	config.UseAst = *useAst
	config.UseFim = *fim
	config.MaxTokens = *maxTokens
	config.Temperature = *temperature
	// Reprocess stop sequences based on the final flag value
	stopSequences := strings.Split(*stop, ",")
	validStopSequences := make([]string, 0, len(stopSequences))
	for _, s := range stopSequences {
		trimmed := strings.TrimSpace(s)
		if trimmed != "" {
			validStopSequences = append(validStopSequences, trimmed)
		}
	}
	// Only override stop if the flag provided non-empty sequences OR if the flag was explicitly passed (even if empty)
	if len(validStopSequences) > 0 || isFlagPassed("stop") {
		config.Stop = validStopSequences
	}

	// Log final config settings being used (optional)
	log.Printf("Final config: %+v", config)

	// --- Determine Input Mode (File vs Prompt Args) ---
	var promptArgs string
	if *filename == "" && flag.NArg() > 0 {
		promptArgs = strings.Join(flag.Args(), " ")
		log.Printf("Using non-flag arguments as prompt: %q", promptArgs)
	}

	// Create a background context. Consider adding timeout later.
	ctx := context.Background()

	// Create a spinner.
	spinner := deepcomplete.NewSpinner()

	// --- Input Validation ---
	if *filename != "" && (*row <= 0 || *col <= 0) {
		// Use exported colors for error message
		log.Fatalf("%sError: Both --row and --col must be positive integers when --file is provided.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
	}
	// Check if we have either file input or prompt input (from args)
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
	// **FIX:** Correctly assign the single return value from NewDeepCompleterWithConfig
	completer, _ := deepcomplete.NewDeepCompleterWithConfig(config)
	// **FIX:** Remove the check for an error that is no longer returned
	// if err != nil {
	// 	log.Fatalf("%sError initializing completer: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
	// }

	// --- Execute Completion Logic ---
	var err error // Declare err here for use in the switch cases

	switch {
	case promptArgs != "": // Direct prompt input from args
		if config.UseFim {
			log.Fatalf("%sError: Fill-in-the-Middle (-fim) requires file input (--file), not a direct prompt.%s", deepcomplete.ColorRed, deepcomplete.ColorReset)
		}
		spinner.Start("Waiting for Ollama...") // Updated spinner message
		var completion string
		// Call the method on the completer instance
		completion, err = completer.GetCompletion(ctx, promptArgs) // Assign to err declared above
		spinner.Stop()
		if err != nil {
			// Use exported colors
			log.Fatalf("%sError getting completion: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
		}

		// Use exported PrettyPrint helper
		// **FIX:** Use exported PrettyPrint
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
				spinner.UpdateMessage("Waiting for Ollama...")
			}()
		}

		// Call the method on the completer instance
		err = completer.GetCompletionStreamFromFile(ctx, *filename, *row, *col, os.Stdout) // Assign to err declared above
		spinner.Stop()                                                                     // Stop spinner after streaming finishes or errors
		if err != nil {
			// Use exported colors
			log.Fatalf("%sError getting completion from file: %v%s", deepcomplete.ColorRed, err, deepcomplete.ColorReset)
		}
		// Output is already streamed, so just exit normally on success
		return

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
