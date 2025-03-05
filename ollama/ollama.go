//go:build linux

package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/host"
	process "github.com/shirou/gopsutil/process"
)

// OllamaClient manages the Ollama process, HTTP communication, output scanning, and resource monitoring.
type OllamaClient struct {
	config                 Config
	process                *os.Process
	processMutex           sync.Mutex
	httpClient             *http.Client
	baseURL                string // Base URL, including host and port
	cpuPriority            int
	monitorIntervalSeconds int
	resourceStatsBuffer    []ResourceStats // Ring buffer for resource stats
	statsBufferIndex       int             // Current index in ring buffer
	statsBufferMaxSize     int             // Max size of the ring buffer
	// commandChan chan string
}

// Config for Ollama client (can be a subset of the main config).
type Config struct {
	Model                  string `yaml:"model"`
	MonitorIntervalSeconds int    `yaml:"monitor_interval_seconds"` // Resource monitor interval in seconds
	OllamaSubprocessPort   int    `yaml:"ollama_subprocess_port"`   // Port for subprocess Ollama server, 0 for dynamic
	OllamaAPIURL           string `yaml:"ollama_api_url"`           // Ollama API base URL
}

// NewClient creates a new OllamaClient.
func NewClient(cfg Config) *OllamaClient {
	return &OllamaClient{
		config: cfg,
		httpClient: &http.Client{
			Timeout: time.Minute, // Example timeout
			Transport: &http.Transport{
				MaxIdleConns:        10,
				IdleConnTimeout:     30 * time.Second,
				DisableKeepAlives:   false, // Enable keep-alive
				MaxIdleConnsPerHost: 10,
			},
		}, // httpClient will use baseURL to make requests
		baseURL:                cfg.OllamaAPIURL,
		cpuPriority:            cfg.CPUPriority, // Initialize cpuPriority from config
		monitorIntervalSeconds: cfg.MonitorIntervalSeconds,
		statsBufferMaxSize:     120,                        // Example: Store up to 120 samples (e.g., 10 minutes of 5-second intervals)
		resourceStatsBuffer:    make([]ResourceStats, 120), // Initialize ring buffer slice
		statsBufferIndex:       0,
	}
}

// EnsureOllamaRunning starts Ollama if it's not already running.
// This is a simplified example and needs more robust error handling and logging.
func (c *OllamaClient) EnsureOllamaRunning(ctx context.Context) error {
	c.processMutex.Lock()
	defer c.processMutex.Unlock()

	if c.process != nil {
		if err := c.process.Signal(syscall.Signal(0)); err == nil {
			return nil // Ollama is already running
		} else {
			slog.Warn("Ollama process appears to be dead, restarting...")
			c.process = nil // Process is no longer valid, needs restart.
		}
	}

	slog.Info("Ensuring Ollama process is running...")

	var cmd *exec.Cmd
	ollamaArgs := []string{"serve"}

	// Handle custom port for subprocess
	port := c.config.OllamaSubprocessPort
	var actualPort int // To store the actual port used (dynamic or configured)
	if port > 0 {
		ollamaArgs = append(ollamaArgs, fmt.Sprintf("--port=%d", port)) // Add --port flag if configured
		actualPort = port
	} else {
		actualPort = 11434 // Default port if dynamic or not configured
	}

	if len(OllamaBinary) > 0 { // Embedded binary
		memfdName := "deepcomplete-ollama"
		memfd, err := syscall.Memfd_create(memfdName, 0)
		if err != nil {
			if port > 0 {
				return fmt.Errorf("memfd_create failed for Ollama subprocess on port %d: %w", port, err)
			} else {
				return fmt.Errorf("memfd_create failed for Ollama subprocess: %w", err)
			}

			return fmt.Errorf("memfd_create failed: %w", err)
		}
		defer syscall.Close(memfd)
		_, err = syscall.Write(memfd, OllamaBinary)
		if err != nil {
			return fmt.Errorf("failed to write Ollama binary to memfd: %w", err)
		}
		if err := syscall.Fchmod(memfd, 0755); err != nil {
			if port > 0 {
				return fmt.Errorf("failed to chmod memfd for Ollama subprocess on port %d: %w", port, err)
			} else {
				return fmt.Errorf("failed to chmod memfd for Ollama subprocess: %w", err)
			}
		}
		memfdPath := fmt.Sprintf("/proc/self/fd/%d", memfd)
		cmd = exec.Command(memfdPath, ollamaArgs...)
	} else { // PATH fallback
		cmd = exec.Command("ollama", ollamaArgs...)
		if runtime.GOOS == "windows" {
			cmd = exec.Command("ollama.exe", ollamaArgs...)
		}
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		if port > 0 {
			return fmt.Errorf("failed to create stderr pipe for Ollama subprocess on port %d: %w", port, err)
		} else {
			return fmt.Errorf("failed to create stderr pipe for Ollama subprocess: %w", err)
		}
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	if err != nil {
		if port > 0 {
			return fmt.Errorf("failed to create stderr pipe for Ollama subprocess on port %d: %w", port, err)
		} else {
			return fmt.Errorf("failed to create stderr pipe for Ollama subprocess: %w", err)
		}
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start Ollama process: %w", err)
	}
	c.process = cmd.Process

	go c.scanOutput(stdoutPipe, slog.LevelInfo)
	go c.scanOutput(stderrPipe, slog.LevelWarn)
	go c.monitorResourceUsage(c.process.Pid)
	// After starting, update baseURL with actual port (dynamic or configured)
	c.baseURL = fmt.Sprintf("http://localhost:%d", actualPort)
	fmt.Printf("Ollama subprocess started on port: %d\n", actualPort) // Print port info for debugging

	time.Sleep(2 * time.Second)
	slog.Info("Ollama process started.", slog.Int("pid", c.process.Pid))

	// Test connection to Ollama API
	if err := c.TestConnection(ctx); err != nil {
		if port > 0 {
			return fmt.Errorf("Ollama subprocess started but API connection failed on port %d: %w", port, err)
		} else {
			return fmt.Errorf("Ollama subprocess started but API connection failed: %w", err)
		}

		return fmt.Errorf("Ollama subprocess started, but API connection test failed: %w", err)
	}
	return nil
}

// StopOllama stops the Ollama process if it's running.
func (c *OllamaClient) StopOllama(ctx context.Context) error {
	c.processMutex.Lock()
	defer c.processMutex.Unlock()

	if c.process != nil {
		slog.Info("Stopping Ollama process...", slog.Int("pid", c.process.Pid))
		if err := c.process.Kill(); err != nil {
			return fmt.Errorf("failed to stop Ollama process: %w", err)
		}
		_, err := c.process.Wait()
		if err != nil {
			slog.Warn("Error waiting for Ollama process to exit (process may have already exited): %v", err)
		} else {
			usage := c.process.ProcessState.SysUsage()
			slog.Info("Ollama process stopped.", slog.Int("pid", c.process.Pid), slog.Any("resource_usage_on_stop", usage))
		}
		c.process = nil
	} else {
		slog.Info("Ollama process is not running, nothing to stop.")
	}
	return nil
}

// RestartOllama restarts the Ollama process.
func (c *OllamaClient) RestartOllama(ctx context.Context) error {
	slog.Info("Restarting Ollama process...")
	if err := c.StopOllama(ctx); err != nil {
		return fmt.Errorf("error stopping Ollama before restart: %w", err)
	}
	return c.EnsureOllamaRunning(ctx)
}

// LoadModel loads a model into Ollama.
func (c *OllamaClient) LoadModel(ctx context.Context, model string) error {
	if model == "" {
		model = c.config.Model // Use default model from config if not specified
	}
	slog.Info("Loading model", slog.String("model", model))

	reqBody, err := json.Marshal(map[string]string{"name": model})
	if err != nil {
		return fmt.Errorf("failed to marshal load model request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/pull", bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create load model request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request to load model failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to load model, status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	// Optionally, stream and process the response body to show progress (like ollama CLI)
	_, _ = io.Copy(os.Stdout, resp.Body) // Example: stream to stdout
	slog.Info("Model loaded successfully", slog.String("model", model))
	return nil
}

// UnloadModel unloads the currently loaded model (if possible via Ollama API).
// Note: Ollama might not have a direct "unload" API. This might involve stopping and restarting
// or relying on Ollama's model management if it unloads models when not in use.
// For now, we'll assume restarting Ollama effectively unloads models.
func (c *OllamaClient) UnloadModel(ctx context.Context) error {
	slog.Info("Unloading model (via Ollama restart)...")
	return c.RestartOllama(ctx) // Simplistic unload by restart
}

// GenerateRequest is the request struct for text generation.
type GenerateRequest struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`
	Stream bool   `json:"stream"` // For future streaming support
}

// GenerateResponse is the response struct for text generation (streaming).
type GenerateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// GenerateTextStream requests streaming text generation from Ollama.
// It returns a channel of strings that emits response chunks and an error channel.
func (c *OllamaClient) GenerateTextStream(ctx context.Context, prompt string) (<-chan string, <-chan error) {
	respChan := make(chan string)
	errChan := make(chan error, 1) // Buffered error channel to avoid blocking

	go func() {
		defer close(respChan)
		defer close(errChan)

		reqData := GenerateRequest{
			Prompt: prompt,
			Model:  c.config.Model,
			Stream: true, // Enable streaming
		}
		reqBody, err := json.Marshal(reqData)
		if err != nil {
			errChan <- fmt.Errorf("failed to marshal generate request: %w", err)
			return
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/generate", bytes.NewBuffer(reqBody))
		if err != nil {
			errChan <- fmt.Errorf("failed to create generate request: %w", err)
			return
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			errChan <- fmt.Errorf("HTTP request to generate text failed: %w", err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			// Optimized error logging: log bodyBytes directly as []byte, format error message without string conversion
			errChan <- fmt.Errorf("generate request failed, status code: %d, body: %w", resp.StatusCode, errors.New("response body"))
			slog.Error("GenerateTextStream failed",
				slog.Int("status_code", resp.StatusCode),
				slog.String("error_detail", fmt.Sprintf("HTTP status %d", resp.StatusCode)), // Optional: more readable error detail
				slog.Any("response_body_bytes", bodyBytes),                                  // Log raw bytes directly
			)
			return
		}

		decoder := json.NewDecoder(resp.Body)
		for {
			var genResp GenerateResponse
			if err := decoder.Decode(&genResp); err != nil {
				if errors.Is(err, io.EOF) {
					return // Stream ended normally
				}
				errChan <- fmt.Errorf("error decoding generate response: %w", err)
				return
			}
			respChan <- genResp.Response // Send response chunk to channel
			if genResp.Done {
				return // Generation is done
			}
		}
	}()

	return respChan, errChan
}

// GenerateText calls GenerateTextStream and aggregates the streamed response into a single string.
// It's kept for backward compatibility but might be deprecated in the future.
func (c *OllamaClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	respChan, errChan := c.GenerateTextStream(ctx, prompt)
	var fullResponse strings.Builder
	for chunk := range respChan {
		fullResponse.WriteString(chunk)
	}
	if err := <-errChan; err != nil {
		return "", err
	}
	return fullResponse.String(), nil
}

// SleepMode unloads the model (for resource saving).
func (c *OllamaClient) SleepMode(ctx context.Context) error {
	slog.Info("Entering sleep mode (unloading model)...")
	return c.UnloadModel(ctx)
}

// WakeUpMode loads the model back (after sleep).
func (c *OllamaClient) WakeUpMode(ctx context.Context) error {
	slog.Info("Waking up (loading model)...")
	return c.LoadModel(ctx, "") // Load default model
}

func (c *OllamaClient) monitorResourceUsage(pid int) {
	proc, err := process.NewProcess(int32(pid))
	if err != nil {
		slog.Error("Failed to get process for resource monitoring", slog.Int("pid", pid), slog.Error(err))
		return
	}

	ticker := time.NewTicker(time.Duration(c.monitorIntervalSeconds) * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		cpuPercent, err := proc.CPUPercent()
		if err != nil {
			slog.Warn("Failed to get CPU percent", slog.Int("pid", pid), slog.Error(err))
			continue
		}
		memInfo, err := proc.MemoryInfo()
		if err != nil {
			slog.Warn("Failed to get memory info", slog.Int("pid", pid), slog.Error(err))
			continue
		}

		gpuUsage := "N/A"
		gpuTemp := "N/A"
		if runtime.GOOS == "linux" {
			gpuPercent, gpuErr := getNvidiaGPUUsage()
			if gpuErr == nil {
				gpuUsage = fmt.Sprintf("%.1f%%", gpuPercent)
			} else {
				slog.Warn("Failed to get NVIDIA GPU usage", slog.Int("pid", pid), slog.Error(gpuErr))
			}

			gpuStatInfo, gpuTempErr := host.SensorsTemperatures()
			if gpuTempErr == nil {
				for _, sensor := range gpuStatInfo {
					if strings.Contains(strings.ToLower(sensor.SensorKey), "gpu") {
						gpuTemp = fmt.Sprintf("%.1f°C", sensor.Temperature)
						break
					}
				}
			} else {
				slog.Warn("Failed to get GPU temperature from sensors", slog.Int("pid", pid), slog.Error(gpuTempErr))
			}
		}

		cpuTemps := "N/A"
		tempStats, tempErr := host.SensorsTemperatures()
		if tempErr == nil {
			var cpuTemperatureReadings []string
			for _, sensor := range tempStats {
				if strings.Contains(strings.ToLower(sensor.SensorKey), "coretemp") || strings.Contains(strings.ToLower(sensor.SensorKey), "cpu") {
					cpuTemperatureReadings = append(cpuTemperatureReadings, fmt.Sprintf("%s:%.1f°C", sensor.SensorKey, sensor.Temperature))
				}
			}
			if len(cpuTemperatureReadings) > 0 {
				cpuTemps = strings.Join(cpuTemperatureReadings, ", ")
			}
		} else {
			slog.Warn("Failed to get CPU/system temperatures", slog.Int("pid", pid), slog.Error(tempErr))
		}

		stats := ResourceStats{ // Create ResourceStats struct
			Timestamp:              time.Now(),
			CPUPercent:             cpuPercent,
			MemoryRSSKB:            memInfo.RSS / 1024,
			MemoryVMSKB:            memInfo.VMS / 1024,
			GpuPercentNvidia:       gpuUsage,
			GpuTemperatureSensor:   gpuTemp,
			CpuTemperaturesSensors: cpuTemps,
		}

		c.resourceStatsBuffer[c.statsBufferIndex] = stats                    // Add stats to ring buffer
		c.statsBufferIndex = (c.statsBufferIndex + 1) % c.statsBufferMaxSize // Increment and wrap index

		slog.Info("Ollama Resource Usage",
			slog.Int("pid", pid),
			slog.Float64("cpu_percent", cpuPercent),
			slog.Uint64("memory_rss_kb", memInfo.RSS/1024),
			slog.Uint64("memory_vms_kb", memInfo.VMS/1024),
			slog.String("gpu_percent_nvidia", gpuUsage),
			slog.String("gpu_temperature_sensor", gpuTemp),
			slog.String("cpu_temperatures_sensors", cpuTemps),
		)
	}
}

// GetResourceStatsHistory returns a copy of the resource stats ring buffer.
func (c *OllamaClient) GetResourceStatsHistory() []ResourceStats {
	history := make([]ResourceStats, c.statsBufferMaxSize)
	c.processMutex.Lock() // Protect access to resourceStatsBuffer
	copy(history, c.resourceStatsBuffer)
	c.processMutex.Unlock()
	return history
}

// TestConnection tests the connection to the Ollama API server.
func (c *OllamaClient) TestConnection(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/version", nil)
	if err != nil {
		return fmt.Errorf("failed to create version request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request to version endpoint failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("version request failed, status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	// Successfully connected
	fmt.Println("Successfully connected to Ollama API at:", c.baseURL) // Connection success message for debugging
	return nil
}
