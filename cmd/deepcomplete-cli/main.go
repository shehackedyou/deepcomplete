package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog" // Use structured logging
	"os"
	"time"

	// NOTE: Replace with your actual module path if different
	"github.com/shehackedyou/deepcomplete"
)

func main() {
	// --- Flag Definitions ---
	filePath := flag.String("file", "", "Path to the Go file (required unless -stdin is used)")
	line := flag.Int("line", 0, "Line number (1-based, required unless -stdin is used)")
	col := flag.Int("col", 0, "Column number (1-based, required unless -stdin is used)")
	stdin := flag.Bool("stdin", false, "Read code snippet from stdin instead of file context")
	// Cycle 1: Add log level flag for overriding config/default
	logLevelFlag := flag.String("log-level", "", "Log level (debug, info, warn, error) - overrides config")
	// Add flags for other config options if needed (e.g., -model, -url)

	flag.Parse()

	// --- Initialize Completer (Loads Config) ---
	// Load config implicitly via NewDeepCompleter.
	// NewDeepCompleter now returns a non-fatal ErrConfig if loading had issues.
	completer, initErr := deepcomplete.NewDeepCompleter()
	if initErr != nil && !errors.Is(initErr, deepcomplete.ErrConfig) {
		// Use a temporary basic logger for fatal init errors before final logger setup
		tempLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
		tempLogger.Error("Fatal error initializing DeepCompleter service", "error", initErr)
		os.Exit(1) // Exit on fatal errors
	}
	if completer == nil {
		// This should ideally not happen if NewDeepCompleter handles errors correctly
		tempLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
		tempLogger.Error("DeepCompleter initialization returned nil unexpectedly")
		os.Exit(1)
	}
	// Log non-fatal config errors later, after the final logger is set up.
	defer func() {
		// Use the final configured logger for shutdown messages
		slog.Info("Closing DeepCompleter service...")
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()

	// --- Setup Logger based on Flag/Config (Cycle 1 Refinement) ---
	initialConfig := completer.GetCurrentConfig()
	chosenLogLevelStr := initialConfig.LogLevel // Start with config level

	// Override with flag if provided
	if *logLevelFlag != "" {
		chosenLogLevelStr = *logLevelFlag
		slog.Debug("Log level overridden by command-line flag", "flag_level", chosenLogLevelStr)
	}

	// Parse the chosen level string
	logLevel, parseLevelErr := deepcomplete.ParseLogLevel(chosenLogLevelStr)
	if parseLevelErr != nil {
		// Use a temporary logger if parsing fails, as default logger isn't set yet
		tempLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
		tempLogger.Warn("Invalid log level specified, using default 'info'", "specified_level", chosenLogLevelStr, "error", parseLevelErr)
		logLevel = slog.LevelInfo // Default to Info
	}

	// Initialize the default slog logger with the determined level
	// Send CLI logs to stderr
	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: false} // Source location less useful for CLI
	logger := slog.New(slog.NewTextHandler(os.Stderr, &handlerOpts))
	slog.SetDefault(logger) // Set as default for the rest of the execution

	// Log initialization confirmation *after* setting the final logger
	slog.Info("DeepComplete service initialized.", "effective_log_level", logLevel.String())
	// Log the non-fatal config error now, if it occurred during init
	if initErr != nil && errors.Is(initErr, deepcomplete.ErrConfig) {
		slog.Warn("DeepCompleter initialized with configuration warnings", "error", initErr)
	}

	// --- Input Validation ---
	// Use the configured slog logger for validation errors now
	if *stdin {
		if *filePath != "" || *line != 0 || *col != 0 {
			slog.Error("Cannot use -file, -line, or -col flags when -stdin is specified.")
			flag.Usage()
			os.Exit(1)
		}
	} else {
		if *filePath == "" {
			slog.Error("Missing required flag: -file")
			flag.Usage()
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
		// Validate file path using helper function
		absPath, pathErr := deepcomplete.ValidateAndGetFilePath(*filePath, slog.Default())
		if pathErr != nil {
			slog.Error("Invalid file path provided via -file flag", "path", *filePath, "error", pathErr)
			os.Exit(1)
		}
		// Check if file exists (optional, but good for CLI)
		if _, statErr := os.Stat(absPath); statErr != nil {
			slog.Error("Cannot access file provided via -file flag", "path", absPath, "error", statErr)
			os.Exit(1)
		}
		// Update filePath to the validated absolute path
		*filePath = absPath
	}

	// --- Execute Command ---
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	if *stdin {
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
		slog.Info("Getting completion from file", "path", *filePath, "line", *line, "col", *col)

		// Use streaming completion, write result directly to stdout
		// Pass a dummy version (0) as CLI doesn't track versions.
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
				slog.Error("Failed to get completion stream from file", "error", completionErr, "file", *filePath, "line", *line, "col", *col)
			}
			fmt.Fprintln(os.Stderr) // Add newline to stderr for errors
			os.Exit(1)
		}
		fmt.Println() // Add newline to stdout for clean separation
	}

	slog.Info("CLI command finished successfully.")
}
