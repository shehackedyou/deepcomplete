package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http" // For pprof
	_ "net/http/pprof"
	"os"
	"time"

	// NOTE: Replace with your actual module path if different
	"github.com/shehackedyou/deepcomplete"
)

// Set at build time
var version = "dev"

func main() {
	// --- Flag Definitions ---
	logLevelFlag := flag.String("log-level", "", "Log level (debug, info, warn, error) - overrides config")
	logFileFlag := flag.String("log-file", "", "Path to log file (optional, defaults to stderr)")
	pprofAddr := flag.String("pprof", "", "Address to expose pprof metrics on (e.g., localhost:6060). Disabled if empty.")

	flag.Parse()

	// --- Setup Logging ---
	var logWriter io.Writer = os.Stderr
	if *logFileFlag != "" {
		f, err := os.OpenFile(*logFileFlag, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0640)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening log file %s: %v\n", *logFileFlag, err)
			os.Exit(1)
		}
		defer f.Close()
		logWriter = f
		fmt.Fprintf(os.Stderr, "Logging to file: %s\n", *logFileFlag)
	}

	// Temporary logger for initialization phase (uses Info level)
	tempHandlerOpts := slog.HandlerOptions{Level: slog.LevelInfo, AddSource: true}
	tempLogger := slog.New(slog.NewTextHandler(logWriter, &tempHandlerOpts))
	tempLogger.Info("Starting DeepComplete LSP", "version", version, "pid", os.Getpid())

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
		slog.Info("Closing DeepCompleter service...") // Use final logger
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

	logLevel, parseLevelErr := deepcomplete.ParseLogLevel(chosenLogLevelStr)
	if parseLevelErr != nil {
		tempLogger.Warn("Invalid log level specified, using default 'info'", "specified_level", chosenLogLevelStr, "error", parseLevelErr)
		logLevel = slog.LevelInfo
	}

	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: true} // Add source info
	finalLogger := slog.New(slog.NewTextHandler(logWriter, &handlerOpts))
	slog.SetDefault(finalLogger) // Set as default for the application

	slog.Info("DeepComplete LSP service components initialized.", "effective_log_level", logLevel.String())
	if initErr != nil && errors.Is(initErr, deepcomplete.ErrConfig) {
		slog.Warn("DeepCompleter initialized with configuration warnings", "error", initErr)
	}

	// --- Check Ollama Availability (Cycle N+5) ---
	// Use a background context for the initial check
	checkCtx, checkCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer checkCancel()
	if err := completer.Client().CheckAvailability(checkCtx, completer.GetCurrentConfig(), finalLogger); err != nil {
		slog.Error("Initial Ollama availability check failed. Completions may not work.", "error", err)
		// Optionally send a showMessage notification if LSP connection were already established,
		// but here it's too early. Log is sufficient.
	} else {
		slog.Info("Initial Ollama availability check successful.")
	}

	// --- Start pprof if enabled ---
	if *pprofAddr != "" {
		slog.Info("Starting pprof HTTP server", "address", *pprofAddr)
		go func() {
			err := http.ListenAndServe(*pprofAddr, nil)
			if err != nil {
				slog.Error("pprof server failed", "error", err)
			}
		}()
	}

	// --- Create and Run LSP Server ---
	server := deepcomplete.NewServer(completer, finalLogger, version)
	server.Run(os.Stdin, os.Stdout)

	slog.Info("DeepComplete LSP server finished.")
}
