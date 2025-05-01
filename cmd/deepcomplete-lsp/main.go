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
	// Setup logging destination *before* initializing slog
	logFile, err := os.OpenFile("deepcomplete-lsp.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0660)
	if err != nil {
		stlog.Fatalf("Failed to open log file: %v", err) // Use standard log for initial fatal error
	}
	defer logFile.Close()

	// --- Setup Temporary Logger for Initialization ---
	// Use a basic stderr logger initially until the final level is determined.
	tempLogger := slog.New(slog.NewTextHandler(io.MultiWriter(os.Stderr, logFile), &slog.HandlerOptions{Level: slog.LevelInfo})) // Default to Info for init

	// --- Initialize Core Service ---
	// This loads configuration internally
	// ** MODIFIED: Cycle 1 Fix - Pass tempLogger to NewDeepCompleter **
	completer, initErr := deepcomplete.NewDeepCompleter(tempLogger)
	if initErr != nil {
		// Log initial error using a temporary basic logger before full slog setup
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
		// Use the final configured logger for shutdown messages
		slog.Info("Closing DeepCompleter service...")
		if err := completer.Close(); err != nil {
			slog.Error("Error closing completer", "error", err)
		}
	}()

	// --- Setup Global Logger ---
	initialConfig := completer.GetCurrentConfig()
	logLevel, parseLevelErr := deepcomplete.ParseLogLevel(initialConfig.LogLevel)
	if parseLevelErr != nil {
		logLevel = slog.LevelInfo // Default to Info
		// Log warning using a temporary logger if parsing fails
		tempLogger.Warn("Invalid log level in config, using default 'info'", "config_level", initialConfig.LogLevel, "error", parseLevelErr)
	}
	logWriter := io.MultiWriter(os.Stderr, logFile)
	handlerOpts := slog.HandlerOptions{Level: logLevel, AddSource: true} // Add source for better debugging
	handler := slog.NewTextHandler(logWriter, &handlerOpts)
	logger := slog.New(handler)
	slog.SetDefault(logger) // Set the configured logger as default

	// Log startup messages using the final logger
	slog.Info("DeepComplete LSP server starting...", "version", appVersion, "log_level", logLevel.String())
	if initErr != nil && errors.Is(initErr, deepcomplete.ErrConfig) {
		// Log config warning again with final logger
		slog.Warn("DeepCompleter initialized with configuration warnings", "error", initErr)
		// LSP Server will send showMessage notification later if possible
	}
	slog.Info("DeepComplete service initialized successfully.")

	// --- Setup Profiling & Metrics ---
	runtime.SetBlockProfileRate(1)
	runtime.SetMutexProfileFraction(1)
	slog.Info("Enabled block and mutex profiling")
	startDebugServer() // Start pprof/expvar HTTP server
	// ** REMOVED call to publishExpvarMetrics() here - it's now handled within NewServer **

	// --- Initialize and Run LSP Server ---
	// Create the LSP server instance from the library
	// NewServer now handles publishing expvar metrics internally
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
		// expvar automatically registers with http.DefaultServeMux,
		// so delegating is sufficient if DefaultServeMux is used.
		// If using a custom mux entirely, use expvar.Handler().
		debugMux.HandleFunc("/debug/vars", expvar.Handler().ServeHTTP)
		if err := http.ListenAndServe(debugListenAddr, debugMux); err != nil {
			// Use the default slog logger as this runs in a separate goroutine
			slog.Error("Debug server failed", "error", err)
		}
	}()
}
