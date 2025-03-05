package main // Package cmd/deepcomplete-cli

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/briandowns/spinner"
	"github.com/shehackedyou/deepcomplete/config" // Import history package
	"github.com/shehackedyou/deepcomplete/ollama"
	"github.com/shehackedyou/deepcomplete/promptformat" // Import promptformat package for prompt formatting
)

// Define emojis for status messages
const (
	EmojiRocket   = "🚀"  // Starting
	EmojiModel    = "󰉯"  // Model related
	EmojiCPU      = "󰻐"  // CPU related
	EmojiGPU      = "󰍹"  // GPU related
	EmojiHistory  = ""  // History related
	EmojiSleep    = "󰒷"  // Sleep mode
	EmojiWake     = "󰒴"  // Wake up mode
	EmojiRestart  = "🔄"  // Restarting
	EmojiGenerate = "✨"  // Generating text
	EmojiSuccess  = "✅"  // Success
	EmojiError    = "❌"  // Error
	EmojiClock    = "⏱️" // Duration timer
	EmojiPercent  = "󰏖"  // Percentage
	EmojiOutput   = "󰅃"  // Output File
	EmojiLog      = "󰇫"  // Log File
)

// SpinnerController encapsulates spinner logic
type SpinnerController struct {
	spinner            *spinner.Spinner
	startTime          time.Time
	statusMessage      string
	percentageComplete string
}

// NewSpinnerController creates a new SpinnerController
func NewSpinnerController(outputWriter io.Writer) *SpinnerController {
	s := spinner.New(spinner.CharSets[9], 100*time.Millisecond, spinner.WithWriter(outputWriter))
	s.Color("fgGreen")
	return &SpinnerController{
		spinner: s,
	}
}

// Start starts the spinner and sets the initial status message
func (sc *SpinnerController) Start(message string) {
	sc.startTime = time.Now()
	sc.statusMessage = message
	sc.updateSpinnerMessage()
	sc.spinner.Start()
}

// Stop stops the spinner and sets the final status message
func (sc *SpinnerController) Stop(success bool, finalMessage string) {
	duration := time.Since(sc.startTime)
	finalStatusMessage := finalMessage

	if finalMessage != "" {
		finalStatusMessage = fmt.Sprintf("%s %s %s", finalMessage, EmojiClock, formatDuration(duration))
	} else {
		finalStatusMessage = fmt.Sprintf("%s %s", EmojiClock, formatDuration(duration))
	}

	if success {
		if finalStatusMessage != "" {
			sc.spinner.FinalMSG = fmt.Sprintf("%s %s\n", finalStatusMessage, EmojiSuccess)
		} else {
			sc.spinner.FinalMSG = fmt.Sprintf("%s\n", EmojiSuccess)
		}
		sc.spinner.StopCharacter = spinner.EmojiCharacters[EmojiSuccess]
	} else {
		if finalMessage != "" {
			sc.spinner.FinalMSG = fmt.Sprintf("%s %s\n", finalMessage, EmojiError)
		} else {
			sc.spinner.FinalMSG = fmt.Sprintf("%s\n", EmojiError)
		}
		sc.spinner.StopCharacter = spinner.EmojiCharacters[EmojiError]
	}
	sc.spinner.Stop()
}

// UpdateMessage updates the spinner status message and percentage (duration is auto-updated)
func (sc *SpinnerController) UpdateMessage(message string, percentage string) {
	sc.statusMessage = message
	sc.percentageComplete = percentage
	sc.updateSpinnerMessage()
}

// UpdateMessageWithDuration updates only the message part of the spinner, duration is still updated
func (sc *SpinnerController) UpdateMessageWithDuration(message string) {
	sc.statusMessage = message
	sc.updateSpinnerMessage()
}

// updateSpinnerMessage constructs and updates the spinner message based on current state
func (sc *SpinnerController) updateSpinnerMessage() {
	statusParts := []string{}
	if sc.statusMessage != "" {
		statusParts = append(statusParts, sc.statusMessage)
	}
	duration := formatDuration(time.Since(sc.startTime))
	if duration != "" && sc.spinner.IsActive() {
		statusParts = append(statusParts, fmt.Sprintf("%s %s", EmojiClock, duration))
	}
	if sc.percentageComplete != "" {
		statusParts = append(statusParts, fmt.Sprintf("%s %s", EmojiPercent, sc.percentageComplete))
	}

	sc.spinner.Suffix = "  " + strings.Join(statusParts, "  ")
}

// App holds the application state and components
type App struct {
	config        config.Config
	statusSpinner *SpinnerController
	multiWriter   io.Writer
	logFileWriter io.Writer
	client        *ollama.OllamaClient
	promptFormat  promptformat.PromptFormat // PromptFormat interface
	cfg           config.Config             // Embed the config
}

// NewApp creates a new App instance
func NewApp(cfg config.Config, multiWriter io.Writer, logFileWriter io.Writer, client *ollama.OllamaClient) *App {
	statusSpinner := NewSpinnerController(multiWriter) // Initialize spinner with multiplexed output
	return &App{
		client:        client,
		config:        cfg,
		statusSpinner: statusSpinner,
		multiWriter:   multiWriter,
		logFileWriter: logFileWriter,
		cfg:           cfg, // Embed the config
	}
}

// formatDuration formats duration to string
func formatDuration(duration time.Duration) string {
	return duration.Round(time.Millisecond).String()
}

// printUsageAndExit prints the usage instructions and exits.
func (app *App) printUsageAndExit() {
	fmt.Fprintln(app.multiWriter, "Usage: deepcomplete-cli [prompt] [flags]")
	fmt.Fprintln(app.multiWriter, "  [prompt] is the text generation prompt (positional argument).\n")
	fmt.Fprintln(app.multiWriter, "  Alternatively, you can use flags to provide input:")                                       // Added alternative input method
	fmt.Fprintf(app.multiWriter, "  -input-file <path> -input-row <num> -input-col <num> -input-length <num>\n")                // Document row/col/length flags
	fmt.Fprintln(app.multiWriter, "  -input-file <path>        Path to file to use as prompt (overrides positional prompt).\n") // Document -input-file
	fmt.Fprintln(app.multiWriter, "Flags for code completion:")                                                                 // Added Flags section for clarity
	flag.PrintDefaults()
	os.Exit(2) // Use exit code 2 for usage error
}

func (app *App) handleShowStatsHistory() { // Handle show stats history command
	history := app.client.GetResourceStatsHistory()

	fmt.Fprintln(app.multiWriter, "--- Resource Usage History ---")
	for _, stats := range history {
		if !stats.Timestamp.IsZero() { // Check if stats are valid (not zero-initialized)
			fmt.Fprintf(app.multiWriter, "Timestamp: %s\n", stats.Timestamp.Format(time.RFC3339))
			fmt.Fprintf(app.multiWriter, "  CPU: %.1f%%\n", stats.CPUPercent)
			fmt.Fprintf(app.multiWriter, "  Memory RSS: %d KB, VMS: %d KB\n", stats.MemoryRSSKB, stats.MemoryVMSKB)
			fmt.Fprintf(app.multiWriter, "  GPU (NVIDIA): %s, Temp: %s\n", stats.GpuPercentNvidia, stats.GpuTemperatureSensor)
			fmt.Fprintf(app.multiWriter, "  CPU Temps: %s\n", stats.CpuTemperaturesSensors)
			fmt.Fprintln(app.multiWriter, "-----------------------")
		}
	}
}

func (app *App) handleShowHistory() { // Handle show history command
	cliHistory := app.historyManager.GetHistory()

	fmt.Fprintf(app.multiWriter, "%s %s --- Prompt and Response History ---\n", EmojiHistory, "History")
	if len(cliHistory) == 0 {
		fmt.Fprintln(app.multiWriter, "No history available.")
		return
	}
	for i := len(cliHistory) - 1; i >= 0; i-- { // Reverse order for newest first
		entry := cliHistory[i]
		fmt.Fprintf(app.multiWriter, "%s %s Entry %d: %s\n", EmojiHistory, "History", i+1, entry.Timestamp.Format(time.RFC3339))
		fmt.Fprintf(app.multiWriter, "  Prompt: %s\n", entry.Prompt)

		// Print only the first line or a truncated response for brevity in CLI history
		responseLines := strings.SplitN(strings.TrimSpace(entry.Response), "\n", 2) // Split into max 2 lines
		responsePreview := responseLines[0]
		if len(responseLines) > 1 || len(responsePreview) > 80 {
			responsePreview = responsePreview[:min(80, len(responsePreview))] + "..." // Truncate long responses
		}

		fmt.Fprintf(app.multiWriter, "  Response: %s\n", responsePreview)
		fmt.Fprintln(app.multiWriter, "-----------------------")
	}
}

// readPromptFromFile reads the entire prompt from the input file.
func (app *App) readPromptFromFile(inputFilePath string) (string, error) {
	promptBytes, err := os.ReadFile(inputFilePath)
	if err != nil {
		return "", fmt.Errorf("failed to read input file: %w", err)
	}
	return string(promptBytes), nil
}

// extractPromptFromFile extracts a partial prompt from the input file based on start row, column, and length.
func (app *App) extractPromptFromFile(inputFilePath string, startRow int, startCol int, length int) (string, error) {
	fileBytes, err := os.ReadFile(inputFilePath)
	if err != nil {
		return "", fmt.Errorf("failed to read input file: %w", err)
	}
	if length <= 0 {
		return "", fmt.Errorf("invalid length: %d, length must be > 0", length)
	}

	fileContent := string(fileBytes)
	lines := strings.Split(fileContent, "\n")
	numLines := len(lines)

	if startRow <= 0 || startRow > len(lines) {
		return "", fmt.Errorf("invalid start row: %d, file has %d lines", startRow, len(lines))
	}
	if startCol <= 0 {
		return "", fmt.Errorf("invalid start column: %d, column must be >= 1", startCol)
	}
	startLine := lines[startRow-1] // Adjust to 0-based indexing

	if startCol-1+length > len(startLine) {
		fmt.Fprintf(os.Stderr, "Warning: specified length exceeds line length, clipping to line end.\n")
		length = len(startLine) - (startCol - 1) // Clip length to remaining line length
		if length < 0 {
			length = 0
		}
	} else if startCol > len(startLine) {
		return "", fmt.Errorf("invalid start column: %d, line %d has length %d", startCol, startRow, len(startLine))
	}

	var extractedPrompt strings.Builder
	extractedPrompt.WriteString(startLine[startCol-1 : min(len(startLine), startCol-1+length)]) // Extract substring with length limit

	return strings.TrimSpace(extractedPrompt.String()), nil // Trim whitespace
}

func (app *App) generateResponse(ctx context.Context, prompt string, prefixFlag string, suffixFlag string, jsonOutputFlag bool, outputFileFlag string) string {
	if err := app.client.EnsureOllamaRunning(ctx); err != nil {
		fmt.Fprintf(app.multiWriter, "%s %s Error ensuring Ollama is running: %v\n", EmojiError, EmojiRocket, err)
		os.Exit(1)
	}

	if prefixFlag != "" || suffixFlag != "" { // Contextual prompt
		prompt = fmt.Sprintf("Prefix:\n%s\n\nSuffix:\n%s\n\nComplete the following code:\n%s", prefixFlag, suffixFlag, prompt)
	}

	app.statusSpinner.Start(fmt.Sprintf("%s %s Generating text...", EmojiGenerate, "Generate"))
	respChan, errChan := app.client.GenerateTextStream(ctx, prompt)

	outputFileWriter := app.multiWriter // Default output writer is multiWriter
	if outputFileFlag != "" {
		outFile, fileErr := os.Create(outputFileFlag)
		if fileErr != nil {
			fmt.Fprintf(app.multiWriter, "%s %s Error creating output file '%s': %v\n", EmojiError, EmojiOutput, outputFileFlag, fileErr)
			os.Exit(1)
		}
		defer outFile.Close()
		outputFileWriter = outFile
		fmt.Fprintf(app.logFileWriter, "%s %s LLM response output is being written to file: %s\n", EmojiOutput, "Output", outputFileFlag)
		fmt.Printf("%s %s LLM response output is being written to file: %s\n", EmojiOutput, "Output", outputFileFlag)
	}

	var fullResponse strings.Builder // Accumulate full response for history
	if jsonOutputFlag {              // JSON output - still basic, streaming JSON could be enhanced
		fmt.Fprintln(outputFileWriter, "{\"response\": \"")
		for chunk := range respChan {
			fmt.Fprint(outputFileWriter, chunk)
			fullResponse.WriteString(chunk) // Accumulate for history even in JSON mode (though not ideal)
		}
		fmt.Fprintln(outputFileWriter, "\"}")
	} else {
		for chunk := range respChan {
			fmt.Fprint(outputFileWriter, chunk)
			fullResponse.WriteString(chunk) // Accumulate full response for history
			app.statusSpinner.UpdateMessageWithDuration(fmt.Sprintf("%s %s Generating text...", EmojiGenerate, "Generate"))
		}
	}
	app.statusSpinner.Stop(<-errChan == nil, fmt.Sprintf("%s %s Generation complete", EmojiGenerate, "Generate"))

	if err := <-errChan; err != nil {
		fmt.Fprintf(app.multiWriter, "%s %s Error generating text: %v\n", EmojiError, EmojiGenerate, err)
		os.Exit(1)
	}
	return fullResponse.String()
}

func (app *App) saveHistory(prompt string, fullResponse string, jsonOutputFlag bool, outputFileFlag string) {
	// Add history entry AFTER successful generation and output
	if !jsonOutputFlag && outputFileFlag == "" { // Only save history for stdout/non-json output
		historyErr := app.historyManager.AddEntry(prompt, fullResponse)
		if historyErr != nil { // Log history saving error, but don't interrupt main operation
			fmt.Fprintf(app.logFileWriter, "%s %s Error saving history: %v\n", EmojiWarning, EmojiHistory, historyErr)
		}
	}
}

// main is the entry point of the deepcomplete-cli application.
func main() {
	// Define command-line flags
	jsonOutputFlag := flag.Bool("json", false, "Output in JSON format (not applicable for streaming yet)")
	modelFlag := flag.String("model", "", "Ollama model to use (overrides config)")
	sleepFlag := flag.Bool("sleep", false, "Enter sleep mode (unload model)")
	wakeFlag := flag.Bool("wake", false, "Wake up from sleep mode (load model)")
	restartFlag := flag.Bool("restart", false, "Restart Ollama server")
	loadModelFlag := flag.String("load", "", "Load a specific model")
	unloadModelFlag := flag.Bool("unload", false, "Unload the current model")
	prefixFlag := flag.String("prefix", "", "Code prefix for contextual prompting")
	suffixFlag := flag.String("suffix", "", "Code suffix for contextual prompting")
	logFileFlag := flag.String("log-file", "", "Path to log file for application logs")
	outputFileFlag := flag.String("output-file", "", "Path to file for LLM response output")
	outputRowFlag := flag.Int("output-row", 0, "Row number in output file to insert response (1-based)")               // Flag for output row
	outputColFlag := flag.Int("output-col", 0, "Column number in output file to insert response (1-based)")            // Flag for output column
	inputEndRowFlag := flag.Int("input-end-row", 0, "End row number in input file for prompt (1-based, inclusive)")    // New flag for input end row
	inputEndColFlag := flag.Int("input-end-col", 0, "End column number in input file for prompt (1-based, inclusive)") // New flag for input end col
	inputRowFlag := flag.Int("input-row", 0, "Row number in input file to start prompt from (1-based)")                // New flag
	inputColFlag := flag.Int("input-col", 0, "Column number in input file to start prompt from (1-based)")             // New flag
	inputLengthFlag := flag.Int("input-length", 0, "Length of prompt from input file (number of characters)")          // New flag for input length
	testConnectionFlag := flag.Bool("test-connection", false, "Test connection to Ollama server and exit")             // New flag
	showHistoryFlag := flag.Bool("show-history", false, "Display command history")                                     // New flag to show history
	showStatsHistoryFlag := flag.Bool("show-stats-history", false, "Display resource usage statistics history")        // New flag

	flag.Parse()

	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %s Error loading config: %v\n", EmojiError, "Config", err)
		os.Exit(1)
	}

	if *modelFlag != "" {
		cfg.Model = *modelFlag // Override model from command line
	}

	clientCfg := ollama.Config{Model: cfg.Model, MonitorIntervalSeconds: cfg.MonitorIntervalSeconds, OllamaAPIURL: cfg.OllamaAPIURL, OllamaSubprocessPort: cfg.OllamaSubprocessPort} // Pass all config values
	client := ollama.NewClient(clientCfg)
	ctx := context.Background()

	// --- Output Multiplexing Setup ---
	outputWriters := []io.Writer{os.Stdout}
	var logFileWriter io.Writer = os.Stdout

	if *logFileFlag != "" {
		logFile, fileErr := os.OpenFile(*logFileFlag, os.O_CREATE|os.O_WRITER|os.O_APPEND, 0644)
		if fileErr != nil {
			fmt.Fprintf(os.Stderr, "%s %s Error opening log file '%s': %v\n", EmojiError, EmojiLog, *logFileFlag, fileErr)
			os.Exit(1)
		}
		defer logFile.Close()
		outputWriters = append(outputWriters, logFile)
		logFileWriter = logFile
		fmt.Printf("%s %s Application logs are being written to file: %s\n", EmojiLog, "Logs", *logFileFlag)
	}

	multiWriter := io.MultiWriter(outputWriters...)

	app := NewApp(cfg, multiWriter, logFileWriter, client) // Create App instance, pass client

	// Determine Prompt Format based on flags
	var promptFmt promptformat.PromptFormat
	if *lineCompletionFlag {
		promptFmt = promptformat.CodeCompletionPromptFormat{}
	} else {
		promptFmt = promptformat.GeneralTextPromptFormat{} // Default to general text format
	}
	app.promptFormat = promptFmt // Set the prompt format in the app

	// Initialize History Manager after App is created, now using configured history file
	err = app.initHistoryManager()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %s Error initializing history manager: %v\n", EmojiError, EmojiHistory, err)
		os.Exit(1)
	}

	if *showStatsHistoryFlag { // Handle show stats history command
		app.handleShowStatsHistory()
		return
	} else if *showHistoryFlag { // Handle show history command
		app.handleShowHistory()
		return
	} else if *testConnectionFlag { // Handle test connection command
		if err := app.handleTestConnection(ctx); err != nil { // Handle potential error from TestConnection
			os.Exit(1) // Exit with error code if test fails
		}
		return // Exit after connection test
	}

	// --- Command Routing based on Flags ---
	// --- CLI Operations are now methods of App ---
	if *sleepFlag {
		app.handleSleep(ctx)
		return
	}

	if *wakeFlag {
		app.handleWake(ctx)
		return
	}

	if *restartFlag {
		app.handleRestart(ctx)
		return
	}

	if *loadModelFlag != "" {
		app.handleLoadModel(ctx, *loadModelFlag)
		return
	}

	if *unloadModelFlag {
		app.handleUnloadModel(ctx)
		return
	}

	prompt := ""
	if flag.NArg() > 0 {
		prompt = flag.Arg(0)
	}

	// --- Input Prompt Handling ---
	// Read prompt from input file if -input-file flag is provided
	if prompt != "" {
		// Extract prompt from input file with row, column, and length specification if flags are provided
		if *inputFileFlag != "" && *inputRowFlag > 0 && *inputColFlag > 0 && *inputLengthFlag > 0 {
			prompt, err = app.extractPromptFromFile(ctx, *inputFileFlag, *inputRowFlag, *inputColFlag, *inputLengthFlag)
			if err != nil {
				fmt.Fprintf(app.multiWriter, "%s %s Error extracting prompt from input file '%s' at %d:%d with length %d: %v\n", EmojiError, EmojiOutput, *inputFileFlag, *inputRowFlag, *inputColFlag, *inputLengthFlag, err)
				app.printUsageAndExit()
				os.Exit(1)
			}
			if prompt == "" { // Check if extracted prompt is empty
				fmt.Fprintf(app.multiWriter, "%s %s Warning: Extracted prompt from input file '%s' at %d:%d with length %d is empty, using empty prompt.\n", EmojiWarning, EmojiOutput, *inputFileFlag, *inputRowFlag, *inputColFlag, *inputLengthFlag)
			}
		} else { // Use positional prompt as is
			app.handleGenerateText(ctx, prompt, *prefixFlag, *suffixFlag, *jsonOutputFlag, *outputFileFlag)
			return
		}

	} else if *inputFileFlag != "" { // If no positional prompt, check for -input-file
		prompt, err := app.readPromptFromFile(*inputFileFlag)
		if err != nil {
			fmt.Fprintf(app.multiWriter, "%s %s Error reading input file '%s': %v\n", EmojiError, EmojiOutput, *inputFileFlag, err)
			app.printUsageAndExit()
			os.Exit(1)
		}
		app.handleGenerateText(ctx, prompt, *prefixFlag, *suffixFlag, *jsonOutputFlag, *outputFileFlag)
		return
	}

	app.printUsageAndExit()
}
