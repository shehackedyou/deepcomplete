// cmd/deepcomplete-cli/main.go
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	deepcomplete "github.com/shehackedyou/deepcomplete"
)

func main() {
	// Define command-line flags.
	ollamaURL := flag.String("ollama-url", deepcomplete.DefaultConfig.OllamaURL, "Ollama API URL")
	model := flag.String("model", deepcomplete.DefaultConfig.Model, "Ollama model to use")
	row := flag.Int("row", 0, "Row number in the file (starting from 1)")
	col := flag.Int("col", 0, "Column number in the file (starting from 1)")
	filename := flag.String("file", "", "File to read code from")
	prompt := flag.String("prompt", "", "Code prompt to complete (overrides file)")
	useLexer := flag.Bool("lexer", deepcomplete.DefaultConfig.UseLexer, "Use lexer to provide context")
	maxTokens := flag.Int("max-tokens", deepcomplete.DefaultConfig.MaxTokens, "Maximum number of tokens in the completion")
	temperature := flag.Float64("temperature", deepcomplete.DefaultConfig.Temperature, "Temperature for the model")
	stop := flag.String("stop", deepcomplete.defaultStop, "Stop sequence for the model (comma-separated)")

	flag.Parse()

	// Basic configuration.
	config := deepcomplete.Config{
		OllamaURL:   *ollamaURL,
		Model:       *model,
		UseLexer:    *useLexer,
		MaxTokens:   *maxTokens,
		Temperature: *temperature,
		Stop:        strings.Split(*stop, ","), // Convert comma-separated string to slice
	}

	// Create a background context.
	ctx := context.Background()

	// Create a spinner.
	spinner := deepcomplete.NewSpinner()

	// Middleware chain (example).
	middleware := []deepcomplete.Middleware{
		deepcomplete.lintResult,
		deepcomplete.removeExternalLibraries,
		// Add more middleware here as needed.
	}

	// Check if both row and col are provided with filename.
	if *filename != "" && (*row == 0 || *col == 0) {
		log.Fatal("Both --row and --col are required when --file is provided")
	}

	// Get completion based on flags.
	var completion string
	var err error

	switch {
	case *prompt != "": // If prompt is provided, use it directly.
		spinner.Start("Getting completion from Ollama...")
		if config.UseLexer {
			completion, err = deepcomplete.GetCompletionWithLexer(ctx, *prompt, config)
		} else {
			completion, err = deepcomplete.GetCompletion(ctx, *prompt, config)
		}
		spinner.Stop()
		if err != nil {
			log.Fatalf("%sError getting completion: %v%s", deepcomplete.colorRed, err, deepcomplete.colorReset)
		}
	case *filename != "": // If filename, row, and col are provided, use them.
		spinner.Start("Getting completion from Ollama...")
		// Use the new function that streams.
		err = deepcomplete.GetCompletionWithComments(ctx, *filename, *row, *col, config, os.Stdout)
		spinner.Stop()
		if err != nil {
			log.Fatalf("%sError getting completion from file: %v%s", deepcomplete.colorRed, err, deepcomplete.colorReset)
		}
		return // Exit after streaming. The output has already been written.

	default:
		log.Fatal("Either --prompt or --file, --row, and --col must be provided")
	}

	// Process the completion with middleware.
	if completion != "" { // only if we have a non-stream completion.
		processedCompletion, err := deepcomplete.ProcessCodeWithMiddleware(completion, middleware...)
		if err != nil {
			log.Fatalf("%sError processing completion: %v%s", deepcomplete.colorRed, err, deepcomplete.colorReset)
		}
		prettyPrint(deepcomplete.colorGreen, "Completed code:\n")
		fmt.Println(processedCompletion)
	}
}

// go run cmd/deepcomplete-cli/main.go --file main.go --row 10 --col 20
// go run cmd/deepcomplete-cli/main.go --prompt "func main() {"
// go run cmd/deepcomplete-cli/main.go --help
// ollama run deepseek-coder-r2
