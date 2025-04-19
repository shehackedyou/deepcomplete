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
	slog.SetDefault(logger) // Set as default for convenience

	// --- Flag Definitions ---
	filePath := flag.String("file", "", "Path to the Go file (required unless -stdin is used)")
	line := flag.Int("line", 0, "Line number (1-based, required unless -stdin is used)")
	col := flag.Int("col", 0, "Column number (1-based, required unless -stdin is used)")
	stdin := flag.Bool("stdin", false, "Read code snippet from stdin instead of file context")
	// Add flags for other config options if needed, e.g.:
	// model := flag.String("model", "", "Ollama model to use (overrides config)")
	// ollamaURL := flag.String("url", "", "Ollama URL (overrides config)")
	// TODO: Add flag to control log level? e.g., -debug

	flag.Parse()

	// --- Input Validation (Defensive Programming) ---
	if *stdin {
		if *filePath != "" || *line != 0 || *col != 0 {
			slog.Error("Cannot use -file, -line, or -col flags when -stdin is specified.")
			os.Exit(1)
		}
	} else {
		if *filePath == "" {
			slog.Error("Missing required flag: -file")
			flag.Usage() // Print usage information
			os.Exit(1)
		}
		if *line <= 0 {
			slog.Error("Invalid value for -line: must be positive", "value", *line)
			flag.Usage()
			os.Exit(1)
		}
		if *col <= 0 {
			slog.Error("Invalid value for -col: must be positive", "value", *col)
			flag.Usage()
			os.Exit(1)
		}
		// Check if file exists (optional, but good for CLI)
		if _, err := os.Stat(*filePath); err != nil {
			slog.Error("Cannot access file provided via -file flag", "path", *filePath, "error", err)
			os.Exit(1)
		}
	}

	// --- Initialize Completer ---
	// Load config implicitly via NewDeepCompleter (it calls LoadConfig)
	completer, err := deepcomplete.NewDeepCompleter()
	if err != nil {
		// Use slog for fatal errors
		slog.Error("Failed to initialize DeepCompleter service", "error", err)
		// If config error, maybe just warn and proceed with defaults if possible?
		// For now, exit on any init error.
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing DeepCompleter service...")
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()
	slog.Info("DeepComplete service initialized.")

	// --- Execute Command ---
	// Add a reasonable timeout for the CLI operation
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
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
		// NOTE: GetCompletionStreamFromFile now expects version, but CLI doesn't track it.
		// Pass a dummy version (e.g., 0 or 1) or modify the core function signature
		// if version is strictly needed even without LSP context (less likely).
		// Let's assume version 0 is acceptable for CLI usage where versioning isn't relevant.
		dummyVersion := 0
		err := completer.GetCompletionStreamFromFile(ctx, *filePath, dummyVersion, *line, *col, os.Stdout)
		if err != nil {
			// Check for specific, potentially user-actionable errors
			if errors.Is(err, context.DeadlineExceeded) {
				slog.Error("Completion request timed out", "file", *filePath, "line", *line, "col", *col)
			} else if errors.Is(err, context.Canceled) {
				slog.Warn("Completion request cancelled", "file", *filePath, "line", *line, "col", *col)
			} else if errors.Is(err, deepcomplete.ErrOllamaUnavailable) {
				slog.Error("Completion backend (Ollama) unavailable", "error", err)
			} else if errors.Is(err, deepcomplete.ErrAnalysisFailed) {
				slog.Error("Code analysis failed (see logs for details)", "error", err)
			} else {
				// Log other internal errors
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
