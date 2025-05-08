// deepcomplete/lsp_handlers_lifecycle.go
// Contains LSP method handlers related to the server lifecycle (initialize, shutdown, exit).
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

	// Define server capabilities
	serverCapabilities := ServerCapabilities{
		TextDocumentSync: &TextDocumentSyncOptions{
			OpenClose: true,
			Change:    TextDocumentSyncKindFull, // Only support full document sync
		},
		CompletionProvider: &CompletionOptions{
			// Define trigger characters, resolve provider, etc. if needed later
		},
		HoverProvider:      true, // Provide hover information
		DefinitionProvider: true, // Provide definition information
		CodeActionProvider: true, // Announce code action capability (Cycle N+2)
		// TODO: Add other capabilities like signatureHelpProvider, referencesProvider etc.
	}

	result := InitializeResult{
		Capabilities: serverCapabilities,
		ServerInfo:   s.serverInfo, // Server info defined in NewServer
	}

	// Store client capabilities for later reference
	s.clientCaps = params.Capabilities
	s.initParams = &params // Store init params if needed later

	logger.Info("Initialization successful", "server_capabilities", result.Capabilities)
	return result, nil
}

// handleShutdown handles the 'shutdown' request.
// The server should prepare for termination but not exit yet.
func (s *Server) handleShutdown(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, logger *slog.Logger) (any, error) {
	logger.Info("Handling shutdown request")
	// Perform any pre-shutdown cleanup if necessary.
	// Core completer cleanup happens when the main process exits.
	return nil, nil
}

// handleExit handles the 'exit' notification.
// The server should terminate its process.
func (s *Server) handleExit(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, logger *slog.Logger) (any, error) {
	logger.Info("Handling exit notification")
	// Closing the connection signals the main Run loop to exit.
	if s.conn != nil {
		s.conn.Close()
	}
	// The main function can handle the actual os.Exit code based on shutdown status if needed.
	return nil, nil
}
