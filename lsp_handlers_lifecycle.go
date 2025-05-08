// deepcomplete/lsp_handlers_lifecycle.go
// Contains LSP method handlers related to the server lifecycle (initialize, shutdown, exit).
package deepcomplete

import (
	"context"
	"log/slog"
	"time" // Import time for the timeout

	"github.com/sourcegraph/jsonrpc2"
)

// ============================================================================
// LSP Lifecycle Method Handlers
// ============================================================================

// handleInitialize handles the 'initialize' request.
// It stores client capabilities and returns server capabilities.
func (s *Server) handleInitialize(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params InitializeParams, logger *slog.Logger) (any, error) {
	clientName := "unknown"
	clientVersion := ""
	if params.ClientInfo != nil {
		clientName = params.ClientInfo.Name
		clientVersion = params.ClientInfo.Version
	}
	logger.Info("Handling initialize request", "client_name", clientName, "client_version", clientVersion)

	// Define server capabilities
	serverCapabilities := ServerCapabilities{
		TextDocumentSync: &TextDocumentSyncOptions{
			OpenClose: true,
			Change:    TextDocumentSyncKindFull, // Only support full document sync
		},
		CompletionProvider: &CompletionOptions{
			TriggerCharacters: []string{".", "(", " "}, // Basic triggers
			ResolveProvider:   false,
		},
		HoverProvider:      true,
		DefinitionProvider: true,
		CodeActionProvider: &CodeActionOptions{
			CodeActionKinds: []CodeActionKind{
				CodeActionKindQuickFix,
			},
			ResolveProvider: false,
		},
	}

	result := InitializeResult{
		Capabilities: serverCapabilities,
		ServerInfo:   s.serverInfo,
	}

	s.clientCaps = params.Capabilities
	s.initParams = &params

	logger.Info("Initialization successful", "server_capabilities", result.Capabilities)
	return result, nil
}

// handleShutdown handles the 'shutdown' request.
// The server should prepare for termination but not exit yet.
func (s *Server) handleShutdown(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, logger *slog.Logger) (any, error) {
	logger.Info("Handling shutdown request")
	// Perform any pre-shutdown cleanup if necessary.
	return nil, nil
}

// handleExit handles the 'exit' notification.
// The server should terminate its process.
func (s *Server) handleExit(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, logger *slog.Logger) (any, error) {
	logger.Info("Handling exit notification")
	if s.conn != nil {
		// Use a short timeout for closing the connection gracefully
		// closeCtx variable removed as it wasn't used. The timeout is implicitly handled by the context passed to Close.
		_, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel() // Ensure cancel is called eventually
		if err := s.conn.Close(); err != nil {
			logger.Warn("Error closing connection on exit", "error", err)
		} else {
			logger.Debug("Connection closed successfully on exit")
		}
	}
	// The main function's DisconnectNotify will unblock, allowing the process to exit.
	return nil, nil
}
