// deepcomplete/lsp_handlers_lifecycle.go
// Contains LSP method handlers related to the server lifecycle (initialize, shutdown, exit).
// Cycle 5: Moved implementations from lsp_server.go.
package deepcomplete

import (
	"context"
	"log/slog"

	"github.com/sourcegraph/jsonrpc2"
)

// ============================================================================
// LSP Lifecycle Method Handlers
// ============================================================================

// handleInitialize handles the 'initialize' request.
// It stores client capabilities and returns server capabilities.
func (s *Server) handleInitialize(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params InitializeParams, logger *slog.Logger) (any, error) {
	logger.Info("Handling initialize request", "client_name", params.ClientInfo.Name, "client_version", params.ClientInfo.Version)

	// Define server capabilities (can be adjusted based on client capabilities if needed)
	serverCapabilities := ServerCapabilities{
		TextDocumentSync: &TextDocumentSyncOptions{
			OpenClose: true,                     // We need open/close notifications
			Change:    TextDocumentSyncKindFull, // We only support full document sync on change
		},
		CompletionProvider: &CompletionOptions{
			// Define trigger characters, resolve provider, etc. if needed later
		},
		HoverProvider:      true, // We provide hover information
		DefinitionProvider: true, // We provide definition information
	}

	result := InitializeResult{
		Capabilities: serverCapabilities,
		ServerInfo:   s.serverInfo, // Server info defined in NewServer
	}

	logger.Info("Initialization successful", "server_capabilities", result.Capabilities)
	return result, nil
}

// handleShutdown handles the 'shutdown' request.
// The server should prepare for termination but not exit yet.
func (s *Server) handleShutdown(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, logger *slog.Logger) (any, error) {
	logger.Info("Handling shutdown request")
	// Perform any pre-shutdown cleanup if necessary (e.g., saving state).
	// Note: The core completer cleanup (like closing DB) happens when the main process exits
	// and its defer function runs, or potentially via a dedicated shutdown method on the completer if needed.
	return nil, nil
}

// handleExit handles the 'exit' notification.
// The server should terminate its process.
func (s *Server) handleExit(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, logger *slog.Logger) (any, error) {
	logger.Info("Handling exit notification")
	// The LSP spec mentions the server should exit with code 0 if shutdown was received, 1 otherwise.
	// For simplicity, we just close the connection, which will terminate the main Run loop.
	// The OS exit code can be handled by the main function if needed.
	if s.conn != nil {
		s.conn.Close() // Closing the connection signals the main loop to exit
	}
	// os.Exit(0) // Or os.Exit(1) depending on shutdown status, but letting Run loop exit is cleaner.
	return nil, nil
}
