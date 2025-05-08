// deepcomplete/lsp_handlers_workspace.go
// Contains LSP method handlers related to workspace events (e.g., configuration changes).
package deepcomplete

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os" // Needed for logger recreation

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

	// Get current config as a base for merging
	currentConfig := s.completer.GetCurrentConfig()
	newConfig := currentConfig // Start with current config
	fileCfg := changedSettings.DeepComplete
	changedKeys := []string{} // Track which keys were actually changed

	// Merge fields only if they were present in the received settings (non-nil pointers in FileConfig)
	if fileCfg.OllamaURL != nil && *fileCfg.OllamaURL != newConfig.OllamaURL {
		newConfig.OllamaURL = *fileCfg.OllamaURL
		changedKeys = append(changedKeys, "OllamaURL")
	}
	if fileCfg.Model != nil && *fileCfg.Model != newConfig.Model {
		newConfig.Model = *fileCfg.Model
		changedKeys = append(changedKeys, "Model")
	}
	if fileCfg.MaxTokens != nil && *fileCfg.MaxTokens != newConfig.MaxTokens {
		newConfig.MaxTokens = *fileCfg.MaxTokens
		changedKeys = append(changedKeys, "MaxTokens")
	}
	// Note: Comparing slices requires more complex logic, assuming any non-nil means change for now
	if fileCfg.Stop != nil {
		newConfig.Stop = *fileCfg.Stop
		changedKeys = append(changedKeys, "Stop")
	}
	if fileCfg.Temperature != nil && *fileCfg.Temperature != newConfig.Temperature {
		newConfig.Temperature = *fileCfg.Temperature
		changedKeys = append(changedKeys, "Temperature")
	}
	logLevelChanged := false
	if fileCfg.LogLevel != nil && *fileCfg.LogLevel != newConfig.LogLevel {
		newConfig.LogLevel = *fileCfg.LogLevel
		changedKeys = append(changedKeys, "LogLevel")
		logLevelChanged = true // Mark that the log level specifically changed
		configLogger.Info("Log level configuration change received", "new_level_setting", newConfig.LogLevel)
	}
	if fileCfg.UseAst != nil && *fileCfg.UseAst != newConfig.UseAst {
		newConfig.UseAst = *fileCfg.UseAst
		changedKeys = append(changedKeys, "UseAst")
	}
	if fileCfg.UseFim != nil && *fileCfg.UseFim != newConfig.UseFim {
		newConfig.UseFim = *fileCfg.UseFim
		changedKeys = append(changedKeys, "UseFim")
	}
	if fileCfg.MaxPreambleLen != nil && *fileCfg.MaxPreambleLen != newConfig.MaxPreambleLen {
		newConfig.MaxPreambleLen = *fileCfg.MaxPreambleLen
		changedKeys = append(changedKeys, "MaxPreambleLen")
	}
	if fileCfg.MaxSnippetLen != nil && *fileCfg.MaxSnippetLen != newConfig.MaxSnippetLen {
		newConfig.MaxSnippetLen = *fileCfg.MaxSnippetLen
		changedKeys = append(changedKeys, "MaxSnippetLen")
	}

	if len(changedKeys) > 0 {
		configLogger.Info("Applying configuration changes from client", "changed_keys", changedKeys)
		// UpdateConfig performs validation internally and updates the completer's config
		if err := s.completer.UpdateConfig(newConfig); err != nil {
			configLogger.Error("Failed to apply updated configuration", "error", err)
			s.sendShowMessage(MessageTypeError, fmt.Sprintf("Failed to apply configuration update: %v", err))
		} else {
			// Update the server's local copy after successful update in completer
			s.config = s.completer.GetCurrentConfig()
			configLogger.Info("Server configuration updated successfully via workspace/didChangeConfiguration")

			// If log level changed, update the server's logger instance
			if logLevelChanged {
				newLevel, parseErr := ParseLogLevel(s.config.LogLevel)
				if parseErr == nil {
					configLogger.Info("Updating server logger level", "new_level", newLevel)
					// Recreate the logger with the new level
					// Assuming TextHandler and output to Stderr for simplicity
					// In a real app, might need more sophisticated handler management
					// TODO: Persist log file handle if logging to file is needed
					newLogger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
						Level:     newLevel,
						AddSource: true, // Keep source consistent
					}))
					s.logger = newLogger // Replace server's logger instance
					// Update the completer's logger as well if it needs the same level
					// This requires the completer to expose a SetLogger method or similar
					// For now, we assume the completer uses the logger passed at creation
					// or slog.Default(), which might not reflect this change immediately
					// if not managed carefully.
					// Consider adding: s.completer.SetLogger(newLogger) if method exists.
					configLogger.Info("Server logger instance updated with new level.")
				} else {
					configLogger.Warn("Cannot update logger level due to parse error", "level_string", s.config.LogLevel, "error", parseErr)
				}
			}
		}
	} else {
		configLogger.Debug("No relevant configuration changes found in workspace/didChangeConfiguration notification")
	}

	return nil, nil
}
