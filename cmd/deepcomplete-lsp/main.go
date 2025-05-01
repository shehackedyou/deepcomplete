package main

import (
	"errors"
	"expvar" // For publishing metrics
	"io"
	stlog "log" // Renamed standard log
	"log/slog"
	"net/http"         // For pprof/expvar server
	_ "net/http/pprof" // Register pprof handlers
	"os"
	"runtime"

	// NOTE: Replace with your actual module path
	"github.com/shehackedyou/deepcomplete"
)

// App version (set via linker flags -ldflags="-X main.appVersion=...")
var appVersion = "dev"

func main() {
	// --- Basic Setup ---
	logFile, err := os.OpenFile("deepcomplete-lsp.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0660)
	if err != nil {
		stlog.Fatalf("Failed to open log file: %v", err) // Use standard log for initial fatal error
	}
	defer logFile.Close()

	// --- Setup Temporary Logger for Initialization ---
	tempLogger := slog.New(slog.NewTextHandler(io.MultiWriter(os.Stderr, logFile), &slog.HandlerOptions{Level: slog.LevelInfo})) // Default to Info for init

	// --- Initialize Core Service ---
	// Pass tempLogger to NewDeepCompleter
	completer, initErr := deepcomplete.NewDeepCompleter(tempLogger)
	if initErr != nil {
		tempLogger.Error("Failed to initialize DeepCompleter service", "error", initErr)
		// Exit on fatal init errors, but allow config warnings to proceed
		if !errors.Is(initErr, deepcomplete.ErrConfig) {
			os.Exit(1)
		}
		if completer == nil { // Ensure completer is non-nil even with config errors
			tempLogger.Error("DeepCompleter initialization returned nil unexpectedly, exiting.")
			os.Exit(1)
		}
	}
	defer func() {
		slog.Info("Closing DeepCompleter service...") // Use final logger
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()

	// --- Setup Global Logger ---
	initialConfig := completer.GetCurrentConfig()
	// Use utility function from deepcomplete package
	logLevel, parseLevelErr := deepcomplete.ParseLogLevel(initialConfig.LogLevel)
	if parseLevelErr != nil {
		logLevel = slog.LevelInfo // Default to Info
		tempLogger.Warn("Invalid log level in config, using default 'info'", "config_level", initialConfig.LogLevel, "error", parseLevelErr)
	}
	logWriter := io.MultiWriter(os.Stderr, logFile)
	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: true} // Add source for debugging
	handler := slog.NewTextHandler(logWriter, &handlerOpts)
	logger := slog.New(handler)
	slog.SetDefault(logger) // Set the configured logger as default

	// Log startup messages using the final logger
	slog.Info("DeepComplete LSP server starting...", "version", appVersion, "log_level", logLevel.String())
	if initErr != nil && errors.Is(initErr, deepcomplete.ErrConfig) {
		slog.Warn("DeepCompleter initialized with configuration warnings", "error", initErr)
	}
	slog.Info("DeepComplete service initialized successfully.")

	// --- Setup Profiling & Metrics ---
	runtime.SetBlockProfileRate(1)
	runtime.SetMutexProfileFraction(1)
	slog.Info("Enabled block and mutex profiling")
	startDebugServer() // Start pprof/expvar HTTP server

	// --- Initialize and Run LSP Server ---
	// Create the LSP server instance
	lspServer := deepcomplete.NewServer(completer, logger, appVersion)

	// Run the server (blocks until shutdown)
	lspServer.Run(os.Stdin, os.Stdout)

	slog.Info("LSP server has shut down gracefully.")
}

// startDebugServer starts the HTTP server for pprof and expvar.
func startDebugServer() {
	debugListenAddr := "localhost:6061" // Consider making configurable
	go func() {
		slog.Info("Starting debug server for pprof/expvar", "addr", debugListenAddr)
		debugMux := http.NewServeMux()
		// Register pprof handlers
		debugMux.HandleFunc("/debug/pprof/", http.DefaultServeMux.ServeHTTP)
		debugMux.HandleFunc("/debug/pprof/cmdline", http.DefaultServeMux.ServeHTTP)
		debugMux.HandleFunc("/debug/pprof/profile", http.DefaultServeMux.ServeHTTP)
		debugMux.HandleFunc("/debug/pprof/symbol", http.DefaultServeMux.ServeHTTP)
		debugMux.HandleFunc("/debug/pprof/trace", http.DefaultServeMux.ServeHTTP)
		// Register expvar handler
		debugMux.HandleFunc("/debug/vars", expvar.Handler().ServeHTTP)
		if err := http.ListenAndServe(debugListenAddr, debugMux); err != nil {
			slog.Error("Debug server failed", "error", err) // Use default slog logger
		}
	}()
}
