package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog" // Cycle 3: Added slog
	"os"
	"time"

	// NOTE: Replace with your actual module path
	"github.com/shehackedyou/deepcomplete"
)

func main() {
	// Cycle 3: Initialize slog for CLI (simple text handler to stderr)
	logLevel := slog.LevelInfo                                            // Default level for CLI, could be flag-controlled
	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: false} // Source location less useful for CLI
	logger := slog.New(slog.NewTextHandler(os.Stderr, &handlerOpts))
	slog.SetDefault(logger)

	// --- Flag Definitions ---
	filePath := flag.String("file", "", "Path to the Go file")
	line := flag.Int("line", 0, "Line number (1-based)")
	col := flag.Int("col", 0, "Column number (1-based)")
	stdin := flag.Bool("stdin", false, "Read code snippet from stdin instead of file")
	// Add flags for other config options if needed, e.g.:
	// model := flag.String("model", "", "Ollama model to use (overrides config)")
	// ollamaURL := flag.String("url", "", "Ollama URL (overrides config)")

	flag.Parse()

	// --- Input Validation ---
	if !*stdin && (*filePath == "" || *line <= 0 || *col <= 0) {
		fmt.Fprintf(os.Stderr, "Usage: %s -file <path> -line <num> -col <num> [flags]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "   or: %s -stdin [flags] < <snippet>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	// --- Initialize Completer ---
	// Load config implicitly via NewDeepCompleter (it calls LoadConfig)
	completer, err := deepcomplete.NewDeepCompleter()
	if err != nil {
		// Use slog for fatal errors
		slog.Error("Failed to initialize DeepCompleter service", "error", err)
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing DeepCompleter service...")
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()
	slog.Info("DeepCompleter service initialized.")

	// --- Execute Command ---
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second) // Add timeout
	defer cancel()

	if *stdin {
		// Read code from stdin
		slog.Info("Reading code snippet from stdin...")
		snippetBytes, err := io.ReadAll(os.Stdin)
		if err != nil {
			slog.Error("Failed to read from stdin", "error", err)
			os.Exit(1)
		}
		snippet := string(snippetBytes)
		slog.Debug("Read snippet", "length", len(snippet))

		// Use basic completion (no file context/analysis)
		completion, err := completer.GetCompletion(ctx, snippet)
		if err != nil {
			slog.Error("Failed to get completion from stdin", "error", err)
			os.Exit(1)
		}
		fmt.Print(completion) // Print result directly to stdout

	} else {
		// Get completion from file context
		slog.Info("Getting completion from file", "path", *filePath, "line", *line, "col", *col)

		// Use streaming completion, write result directly to stdout
		err := completer.GetCompletionStreamFromFile(ctx, *filePath, *line, *col, os.Stdout)
		if err != nil {
			// Check if error is context cancellation or timeout
			if errors.Is(err, context.DeadlineExceeded) {
				slog.Error("Completion request timed out", "file", *filePath, "line", *line, "col", *col)
			} else if errors.Is(err, context.Canceled) {
				slog.Warn("Completion request cancelled", "file", *filePath, "line", *line, "col", *col)
			} else {
				// Log specific error types if needed (e.g., Ollama unavailable vs analysis error)
				slog.Error("Failed to get completion stream from file", "error", err, "file", *filePath, "line", *line, "col", *col)
			}
			// Add a newline to stderr for errors to separate from potential stdout output
			fmt.Fprintln(os.Stderr)
			os.Exit(1)
		}
		// Add a newline to stdout if completion was successful and didn't end with one
		// (The stream might handle this, but ensures clean separation)
		// Note: This might add an extra newline if the completion already ends with one.
		// A more robust way would be to capture the output and check.
		fmt.Println()
	}

	slog.Info("CLI command finished successfully.")
}

// Helper function to truncate strings for logging (if needed, maybe move to utils)
func firstN(s string, n int) string {
	if len(s) > n {
		if n < 0 {
			n = 0
		}
		i := 0
		for j := range s {
			if i == n {
				return s[:j] + "..."
			}
			i++
		}
		return s
	}
	return s
}
