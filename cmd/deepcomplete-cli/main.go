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
	// --- Flag Definitions ---
	filePath := flag.String("file", "", "Path to the Go file (required unless -stdin is used)")
	line := flag.Int("line", 0, "Line number (1-based, required unless -stdin is used)")
	col := flag.Int("col", 0, "Column number (1-based, required unless -stdin is used)")
	stdin := flag.Bool("stdin", false, "Read code snippet from stdin instead of file context")
	// ** Cycle 1: Add log level flag **
	logLevelFlag := flag.String("log-level", "", "Log level (debug, info, warn, error) - overrides config")
	// Add flags for other config options if needed, e.g.:
	// model := flag.String("model", "", "Ollama model to use (overrides config)")
	// ollamaURL := flag.String("url", "", "Ollama URL (overrides config)")

	flag.Parse()

	// --- Initialize Completer (Loads Config) ---
	// Load config implicitly via NewDeepCompleter (it calls LoadConfig)
	// We need the config *before* setting up the final logger
	completer, err := deepcomplete.NewDeepCompleter()
	if err != nil {
		// Use a temporary basic logger for fatal init errors
		tempLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
		tempLogger.Error("Failed to initialize DeepCompleter service", "error", err)
		os.Exit(1)
	}
	defer func() {
		// Use the final configured logger for shutdown messages
		slog.Info("Closing DeepCompleter service...")
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()

	// --- Setup Logger based on Flag/Config (Cycle 1) ---
	initialConfig := completer.GetCurrentConfig()
	chosenLogLevelStr := initialConfig.LogLevel // Start with config level

	// Override with flag if provided
	if *logLevelFlag != "" {
		chosenLogLevelStr = *logLevelFlag
	}

	// Parse the chosen level string
	logLevel, parseLevelErr := deepcomplete.ParseLogLevel(chosenLogLevelStr)
	if parseLevelErr != nil {
		// Log warning using a temporary logger if parsing fails
		tempLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
		tempLogger.Warn("Invalid log level specified, using default 'info'", "specified_level", chosenLogLevelStr, "error", parseLevelErr)
		logLevel = slog.LevelInfo // Default to Info
	}

	// Initialize the default slog logger with the determined level
	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: false} // Source location less useful for CLI
	logger := slog.New(slog.NewTextHandler(os.Stderr, &handlerOpts))
	slog.SetDefault(logger) // Set as default for the rest of the execution

	// Log initialization confirmation *after* setting the final logger
	slog.Info("DeepComplete service initialized.", "effective_log_level", logLevel.String())

	// --- Input Validation (Defensive Programming) ---
	// Use the configured logger (slog) for validation errors now
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

	// --- Execute Command ---
	// Add a reasonable timeout for the CLI operation
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	if *stdin {
		// Read code from stdin
		slog.Info("Reading code snippet from stdin...")
		snippetBytes, readErr := io.ReadAll(os.Stdin)
		if readErr != nil {
			slog.Error("Failed to read from stdin", "error", readErr)
			os.Exit(1)
		}
		snippet := string(snippetBytes)
		slog.Debug("Read snippet", "length", len(snippet))

		// Use basic completion (no file context/analysis)
		completion, completionErr := completer.GetCompletion(ctx, snippet)
		if completionErr != nil {
			slog.Error("Failed to get completion from stdin", "error", completionErr)
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
		completionErr := completer.GetCompletionStreamFromFile(ctx, *filePath, dummyVersion, *line, *col, os.Stdout)
		if completionErr != nil {
			// Check for specific, potentially user-actionable errors
			if errors.Is(completionErr, context.DeadlineExceeded) {
				slog.Error("Completion request timed out", "file", *filePath, "line", *line, "col", *col)
			} else if errors.Is(completionErr, context.Canceled) {
				slog.Warn("Completion request cancelled", "file", *filePath, "line", *line, "col", *col)
			} else if errors.Is(completionErr, deepcomplete.ErrOllamaUnavailable) {
				slog.Error("Completion backend (Ollama) unavailable", "error", completionErr)
			} else if errors.Is(completionErr, deepcomplete.ErrAnalysisFailed) {
				slog.Error("Code analysis failed (see logs for details)", "error", completionErr)
			} else {
				// Log other internal errors
				slog.Error("Failed to get completion stream from file", "error", completionErr, "file", *filePath, "line", *line, "col", *col)
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
