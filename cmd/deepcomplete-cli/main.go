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

// Set at build time
var version = "dev"

func main() {
	// --- Flag Definitions ---
	filePath := flag.String("file", "", "Path to the Go file (required unless -stdin is used)")
	line := flag.Int("line", 0, "Line number (1-based, required unless -stdin is used)")
	col := flag.Int("col", 0, "Column number (1-based, required unless -stdin is used)")
	stdin := flag.Bool("stdin", false, "Read code snippet from stdin instead of file context")
	logLevelFlag := flag.String("log-level", "", "Log level (debug, info, warn, error) - overrides config")
	// Add flags for other config options if needed (e.g., -model, -url)

	flag.Parse()

	// --- Setup Temporary Logger for Initialization ---
	tempLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	// --- Initialize Completer (Loads Config) ---
	completer, initErr := deepcomplete.NewDeepCompleter(tempLogger) // Pass temp logger
	if initErr != nil && !errors.Is(initErr, deepcomplete.ErrConfig) {
		tempLogger.Error("Fatal error initializing DeepCompleter service", "error", initErr)
		os.Exit(1) // Exit on fatal errors
	}
	if completer == nil {
		tempLogger.Error("DeepCompleter initialization returned nil unexpectedly")
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing DeepCompleter service...") // Use final logger (set below)
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()

	// --- Setup Final Logger based on Flag/Config ---
	initialConfig := completer.GetCurrentConfig()
	chosenLogLevelStr := initialConfig.LogLevel

	if *logLevelFlag != "" {
		chosenLogLevelStr = *logLevelFlag
		tempLogger.Debug("Log level overridden by command-line flag", "flag_level", chosenLogLevelStr)
	}

	logLevel, parseLevelErr := deepcomplete.ParseLogLevel(chosenLogLevelStr) // Util func
	if parseLevelErr != nil {
		tempLogger.Warn("Invalid log level specified, using default 'info'", "specified_level", chosenLogLevelStr, "error", parseLevelErr)
		logLevel = slog.LevelInfo // Default to Info
	}

	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: false} // Keep CLI logs concise
	finalLogger := slog.New(slog.NewTextHandler(os.Stderr, &handlerOpts))
	slog.SetDefault(finalLogger) // Set as default

	slog.Info("DeepComplete service initialized.", "effective_log_level", logLevel.String())
	if initErr != nil && errors.Is(initErr, deepcomplete.ErrConfig) {
		slog.Warn("DeepCompleter initialized with configuration warnings", "error", initErr)
	}

	// --- Input Validation ---
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
		// Validate file path using utility function, pass the final logger
		absPath, pathErr := deepcomplete.ValidateAndGetFilePath(*filePath, finalLogger) // Util func
		if pathErr != nil {
			slog.Error("Invalid file path provided via -file flag", "path", *filePath, "error", pathErr)
			os.Exit(1)
		}
		if _, statErr := os.Stat(absPath); statErr != nil {
			slog.Error("Cannot access file provided via -file flag", "path", absPath, "error", statErr)
			os.Exit(1)
		}
		*filePath = absPath // Update to validated absolute path
	}

	// --- Execute Command ---
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
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

		// GetCompletion uses the logger configured in the completer
		completion, completionErr := completer.GetCompletion(ctx, snippet)
		if completionErr != nil {
			slog.Error("Failed to get completion from stdin", "error", completionErr)
			os.Exit(1)
		}
		fmt.Print(completion) // Print result directly to stdout

	} else {
		slog.Info("Getting completion from file context", "path", *filePath, "line", *line, "col", *col) // More descriptive log

		dummyVersion := 0 // CLI doesn't track versions
		// GetCompletionStreamFromFile uses the logger configured in the completer
		completionErr := completer.GetCompletionStreamFromFile(ctx, *filePath, dummyVersion, *line, *col, os.Stdout)
		if completionErr != nil {
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
