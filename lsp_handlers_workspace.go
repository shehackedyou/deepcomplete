// deepcomplete/lsp_handlers_workspace.go
// Contains LSP method handlers related to workspace events (e.g., configuration changes).
// Cycle 5: Moved implementations from lsp_server.go.
package deepcomplete

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/sourcegraph/jsonrpc2"
)

// ============================================================================
// LSP Workspace Method Handlers
// ============================================================================

// handleDidChangeConfiguration handles configuration changes from the client.
// It attempts to parse the relevant section of the settings and updates the server's configuration.
func (s *Server) handleDidChangeConfiguration(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request, params DidChangeConfigurationParams, logger *slog.Logger) (any, error) {
	configLogger := logger.With("req_id", req.ID) // Add req_id for tracing
	configLogger.Info("Handling workspace/didChangeConfiguration")

	// Define a struct that mirrors the expected nested structure from the client.
	// Adjust the "deepcomplete" key if your client sends it differently.
	var changedSettings struct {
		DeepComplete FileConfig `json:"deepcomplete"` // Use FileConfig from deepcomplete_types.go
	}

	// Attempt to unmarshal the *entire* settings object into our struct.
	if err := json.Unmarshal(params.Settings, &changedSettings); err != nil {
		configLogger.Error("Failed to unmarshal workspace/didChangeConfiguration settings", "error", err, "raw_settings", string(params.Settings))
		// Attempt to unmarshal directly into FileConfig if nesting fails
		// This handles clients that might send the settings flatly.
		var directFileCfg FileConfig
		if directErr := json.Unmarshal(params.Settings, &directFileCfg); directErr == nil {
			configLogger.Info("Successfully unmarshalled settings directly into FileConfig (no 'deepcomplete' nesting)")
			changedSettings.DeepComplete = directFileCfg // Use the directly unmarshalled config
		} else {
			configLogger.Error("Also failed to unmarshal settings directly into FileConfig", "direct_error", directErr)
			// Don't return error for notification, but log it.
			return nil, nil
		}
	}

	// Get current config as a base for merging
	newConfig := s.completer.GetCurrentConfig()
	fileCfg := changedSettings.DeepComplete // Use the potentially nested or direct config
	mergedFields := 0

	// Merge fields only if they were present in the received settings (non-nil pointers in FileConfig)
	if fileCfg.OllamaURL != nil {
		newConfig.OllamaURL = *fileCfg.OllamaURL
		mergedFields++
	}
	if fileCfg.Model != nil {
		newConfig.Model = *fileCfg.Model
		mergedFields++
	}
	if fileCfg.MaxTokens != nil {
		newConfig.MaxTokens = *fileCfg.MaxTokens
		mergedFields++
	}
	if fileCfg.Stop != nil {
		newConfig.Stop = *fileCfg.Stop
		mergedFields++
	}
	if fileCfg.Temperature != nil {
		newConfig.Temperature = *fileCfg.Temperature
		mergedFields++
	}
	if fileCfg.LogLevel != nil {
		newConfig.LogLevel = *fileCfg.LogLevel
		mergedFields++
		configLogger.Info("Log level configuration change received", "new_level_setting", newConfig.LogLevel)
	}
	if fileCfg.UseAst != nil {
		newConfig.UseAst = *fileCfg.UseAst
		mergedFields++
	}
	if fileCfg.UseFim != nil {
		newConfig.UseFim = *fileCfg.UseFim
		mergedFields++
	}
	if fileCfg.MaxPreambleLen != nil {
		newConfig.MaxPreambleLen = *fileCfg.MaxPreambleLen
		mergedFields++
	}
	if fileCfg.MaxSnippetLen != nil {
		newConfig.MaxSnippetLen = *fileCfg.MaxSnippetLen
		mergedFields++
	}

	if mergedFields > 0 {
		configLogger.Info("Applying configuration changes from client", "fields_merged", mergedFields)
		// UpdateConfig performs validation internally and updates the completer's config
		if err := s.completer.UpdateConfig(newConfig); err != nil {
			configLogger.Error("Failed to apply updated configuration", "error", err)
			// Notify the client about the failure
			s.sendShowMessage(MessageTypeError, fmt.Sprintf("Failed to apply configuration update: %v", err))
		} else {
			// Update the server's local copy after successful update in completer
			s.config = s.completer.GetCurrentConfig()
			configLogger.Info("Server configuration updated successfully via workspace/didChangeConfiguration")

			// Attempt to update the server's logger level if it changed
			newLevel, parseErr := ParseLogLevel(s.config.LogLevel) // Util func
			if parseErr == nil {
				configLogger.Info("Attempting to update server logger level (implementation specific)", "new_level", newLevel)
				// NOTE: This requires the server's logger to be mutable or recreated.
				// If s.logger is just a copy, this won't work as intended without
				// a mechanism to update the actual logger used by the server instance.
				// For now, this only logs the intent. A better approach might involve
				// having a dynamic level handler or recreating the logger.
			} else {
				configLogger.Warn("Cannot update logger level due to parse error", "level_string", s.config.LogLevel, "error", parseErr)
			}
		}
	} else {
		configLogger.Debug("No relevant configuration changes found in workspace/didChangeConfiguration notification")
	}

	// Notifications don't have responses
	return nil, nil
}
