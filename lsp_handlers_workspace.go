// deepcomplete/lsp_handlers_workspace.go
// Contains LSP method handlers related to workspace events (e.g., configuration changes).
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
	configLogger := logger.With("req_id", req.ID)
	configLogger.Info("Handling workspace/didChangeConfiguration")

	var changedSettings struct {
		DeepComplete FileConfig `json:"deepcomplete"`
	}

	if err := json.Unmarshal(params.Settings, &changedSettings); err != nil {
		configLogger.Error("Failed to unmarshal workspace/didChangeConfiguration settings", "error", err, "raw_settings", string(params.Settings))
		var directFileCfg FileConfig
		if directErr := json.Unmarshal(params.Settings, &directFileCfg); directErr == nil {
			configLogger.Info("Successfully unmarshalled settings directly into FileConfig (no 'deepcomplete' nesting)")
			changedSettings.DeepComplete = directFileCfg
		} else {
			configLogger.Error("Also failed to unmarshal settings directly into FileConfig", "direct_error", directErr)
			return nil, nil
		}
	}

	newConfig := s.completer.GetCurrentConfig()
	fileCfg := changedSettings.DeepComplete
	mergedFields := 0

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
		if err := s.completer.UpdateConfig(newConfig); err != nil {
			configLogger.Error("Failed to apply updated configuration", "error", err)
			s.sendShowMessage(MessageTypeError, fmt.Sprintf("Failed to apply configuration update: %v", err))
		} else {
			s.config = s.completer.GetCurrentConfig() // Update server's local copy
			configLogger.Info("Server configuration updated successfully via workspace/didChangeConfiguration")

			// Pass configLogger to ParseLogLevel (although it doesn't use it)
			newLevel, parseErr := ParseLogLevel(s.config.LogLevel)
			if parseErr == nil {
				configLogger.Info("Attempting to update server logger level (implementation specific)", "new_level", newLevel)
				// Actual logger update logic would go here if the logger handler supports dynamic levels.
			} else {
				configLogger.Warn("Cannot update logger level due to parse error", "level_string", s.config.LogLevel, "error", parseErr)
			}
		}
	} else {
		configLogger.Debug("No relevant configuration changes found in workspace/didChangeConfiguration notification")
	}

	return nil, nil
}
