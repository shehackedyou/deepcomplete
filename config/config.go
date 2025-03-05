package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config holds the deepcomplete configuration.
type Config struct {
	Model                  string `yaml:"model"`
	MonitorIntervalSeconds int    `yaml:"monitor_interval_seconds"` // Resource monitor interval in seconds
	OllamaSubprocessPort   int    `yaml:"ollama_subprocess_port"`   // Port for subprocess Ollama server, 0 for dynamic
	OllamaAPIURL           string `yaml:"ollama_api_url"`           // Ollama API base URL
	HistoryFile            string `yaml:"history_file"`             // Path to the history file
	CPUPriority            int    `yaml:"cpu_priority,omitempty"`   // CPU priority/nice value (optional)
}

// DefaultConfig returns a Config struct with default values.
func DefaultConfig() Config {
	return Config{
		Model:                  "deepseek-coder-r2",
		MonitorIntervalSeconds: 5,                        // Default interval: 5 seconds
		OllamaSubprocessPort:   0,                        // Default subprocess port: dynamic
		OllamaAPIURL:           "http://localhost:11434", // Default Ollama API URL
		HistoryFile:            "",                       // Default history file path will be set in main.go
		CPUPriority:            0,                        // Default CPU priority: no change
	}
}

// LoadConfig loads the configuration from a YAML file.
// If no config file is found, it creates one with default settings.
// It checks in the following order:
// 1. ~/.config/deepcomplete/config.yml
// 2. ~/.deepcomplete/config.yml
// 3. ./.deepcomplete/config.yml
func LoadConfig() (Config, error) {
	cfg := DefaultConfig()

	configPaths := []string{
		filepath.Join(os.Getenv("HOME"), ".config", "deepcomplete", "config.yml"),
		filepath.Join(os.Getenv("HOME"), ".deepcomplete", "config.yml"),
		"./.deepcomplete/config.yml",
	}

	var configPath string
	configFileExists := false
	for _, path := range configPaths {
		if _, err := os.Stat(path); err == nil {
			configPath = path
			configFileExists = true
			break
		}
		if configPath == "" && os.IsNotExist(err) {
			if configPath == "" {
				configPath = path // Take the first path that doesn't exist as the creation target
			}
		}
	}

	if !configFileExists {
		fmt.Printf("Config file not found, creating default config at: %s\n", configPath)
		if err := createDefaultConfig(configPath, cfg); err != nil {
			return Config{}, fmt.Errorf("failed to create default config file: %w", err)
		}
		return cfg, nil // Return default config after creation
	}

	configFile, err := os.Open(configPath)
	if err != nil {
		return Config{}, fmt.Errorf("failed to open config file: %w", err)
	}
	defer configFile.Close()

	decoder := yaml.NewDecoder(configFile)
	if err := decoder.Decode(&cfg); err != nil {
		return Config{}, fmt.Errorf("failed to decode config file: %w", err)
	}

	return cfg, nil
}

func createDefaultConfig(configPath string, defaultCfg Config) error {
	dirPath := filepath.Dir(configPath)
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		if err := os.MkdirAll(dirPath, 0755); err != nil {
			return fmt.Errorf("failed to create config directory: %w", err)
		}
	}

	configFile, err := os.Create(configPath)
	if err != nil {
		return fmt.Errorf("failed to create config file: %w", err)
	}
	defer configFile.Close()

	encoder := yaml.NewEncoder(configFile)
	encoder.SetIndent(2) // Set indentation for readability
	if err := encoder.Encode(defaultCfg); err != nil {
		return fmt.Errorf("failed to encode default config to file: %w", err)
	}

	return nil
}
