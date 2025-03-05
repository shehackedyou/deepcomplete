package history

import (
	"container/ring"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// HistoryEntry represents a single entry in the history
type HistoryEntry struct {
	Timestamp time.Time `json:"timestamp"`
	Prompt    string    `json:"prompt"`
	Response  string    `json:"response"`
}

// HistoryManager manages the prompt and response history
type HistoryManager struct {
	buffer      *ring.Ring
	maxSize     int
	historyFile string
}

// NewHistoryManager creates a new HistoryManager with a given max size and history file path
func NewHistoryManager(maxSize int, historyFile string) (*HistoryManager, error) {
	hm := &HistoryManager{
		buffer:      ring.New(maxSize),
		maxSize:     maxSize,
		historyFile: historyFile,
	}
	if err := hm.loadHistory(); err != nil {
		return nil, err
	}
	return hm, nil
}

// AddEntry adds a new history entry to the ring buffer and saves to file
func (hm *HistoryManager) AddEntry(prompt string, response string) error {
	entry := HistoryEntry{
		Timestamp: time.Now(),
		Prompt:    prompt,
		Response:  response,
	}
	hm.buffer.Value = entry
	hm.buffer = hm.buffer.Next()
	return hm.saveHistory()
}

// GetHistory returns a slice of history entries from the ring buffer
func (hm *HistoryManager) GetHistory() []HistoryEntry {
	history := make([]HistoryEntry, 0, hm.maxSize)
	hm.buffer.Do(func(val interface{}) {
		if entry, ok := val.(HistoryEntry); ok && !entry.Timestamp.IsZero() {
			history = append(history, entry)
		}
	})
	return history
}

// loadHistory loads history from the history file
func (hm *HistoryManager) loadHistory() error {
	if _, err := os.Stat(hm.historyFile); os.IsNotExist(err) {
		return nil // No history file found, nothing to load
	}

	file, err := os.Open(hm.historyFile)
	if err != nil {
		return fmt.Errorf("failed to open history file: %w", err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	var historyEntries []HistoryEntry
	if err := decoder.Decode(&historyEntries); err != nil {
		// Log warning but don't return error to avoid blocking app startup
		fmt.Fprintf(os.Stderr, "Warning: Failed to decode history file %s, history will be empty. Error: %v\n", hm.historyFile, err)
		return nil // Non-fatal error, proceed with empty history
	}

	// Populate ring buffer from loaded history, in reverse order to maintain chronological order in display
	for i := len(historyEntries) - 1; i >= 0; i-- {
		hm.buffer.Value = historyEntries[i]
		hm.buffer = hm.buffer.Next()
	}

	return nil
}

// saveHistory saves history to the history file
func (hm *HistoryManager) saveHistory() error {
	historyToSave := hm.GetHistory() // Get current history from ring buffer

	dirPath := filepath.Dir(hm.historyFile)
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		if err := os.MkdirAll(dirPath, 0755); err != nil {
			return fmt.Errorf("failed to create history directory: %w", err)
		}
	}

	file, err := os.Create(hm.historyFile)
	if err != nil {
		return fmt.Errorf("failed to create history file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent(2) // Pretty print JSON
	if err := encoder.Encode(historyToSave); err != nil {
		return fmt.Errorf("failed to encode history to file: %w", err)
	}
	return nil
}
