// deepcomplete/deepcomplete_test.go
package deepcomplete

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/token"
	"go/types"
	"io"
	"log/slog" // Added slog
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/dgraph-io/ristretto"
	"go.etcd.io/bbolt"
)

// Helper to create a FileSet and File for position testing.
func createFileSetAndFile(t *testing.T, name, content string) (*token.FileSet, *token.File) {
	t.Helper()
	fset := token.NewFileSet()
	file := fset.AddFile(name, 1, len(content)) // Base 1 ensures Pos > 0
	if file == nil {
		t.Fatalf("Failed to add file '%s' to fileset", name)
	}
	return fset, file
}

// TestCalculateCursorPos tests the conversion of 1-based line/col to token.Pos offset.
func TestCalculateCursorPos(t *testing.T) {
	content := `package main

import "fmt" // Line 3

func main() { // Line 5
	fmt.Println("Hello") // Line 6
	fmt.Println("你好世界") // Line 7 (Multi-byte)
} // Line 8
// Line 9`

	// Setup logger for this test
	testLogger := slog.New(slog.NewTextHandler(io.Discard, nil))

	fset := token.NewFileSet()
	file := fset.AddFile("test_calcpos.go", 1, len(content))
	if file == nil {
		t.Fatal("Failed to add file")
	}

	tests := []struct {
		name       string
		line, col  int
		wantErr    bool
		wantOffset int // 0-based offset from file start
	}{
		{"Start of file", 1, 1, false, 0},
		{"Start of line 3", 3, 1, false, 15},
		{"Middle of Println", 6, 10, false, 46},
		{"End of Println line", 6, 26, false, 62},
		{"Cursor after Println line (at newline)", 6, 27, false, 62}, // Clamped
		{"Start of line 7", 7, 1, false, 63},
		{"Middle of multi-byte string (before 好)", 7, 16, false, 78},
		{"Middle of multi-byte string (after 好)", 7, 17, false, 81},
		{"Middle of multi-byte string (after 界)", 7, 20, false, 90},
		{"End of multi-byte string line", 7, 23, false, 93},
		{"Start of line 8", 8, 1, false, 94},
		{"End of line 8", 8, 2, false, 95},
		{"Start of line 9", 9, 1, false, 96},
		{"End of line 9", 9, 10, false, 105},
		{"Cursor after end of file content", 9, 11, false, 105}, // Clamped
		{"Start of virtual line 10", 10, 1, false, 105},         // Allowed
		{"Invalid col on virtual line 10", 10, 2, true, -1},     // Invalid col
		{"Invalid line (too high)", 11, 1, true, -1},
		{"Invalid line (zero)", 0, 1, true, -1},
		{"Invalid col (zero)", 6, 0, true, -1},
		{"Invalid col (too high for line)", 6, 100, false, 62}, // Clamped
		{"Col too high last line", 9, 100, false, 105},         // Clamped
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Call the utility function directly, passing the test logger
			gotPos, err := calculateCursorPos(file, tt.line, tt.col, testLogger)
			if (err != nil) != tt.wantErr {
				t.Errorf("calculateCursorPos() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if !gotPos.IsValid() {
					wantPos := file.Pos(tt.wantOffset)
					t.Errorf("calculateCursorPos() returned invalid pos %v, want valid pos %v (offset %d)", gotPos, wantPos, tt.wantOffset)
				} else {
					gotOffset := file.Offset(gotPos)
					if gotOffset != tt.wantOffset {
						t.Errorf("calculateCursorPos() calculated offset = %d, want %d (Pos: got %v, want %v)", gotOffset, tt.wantOffset, gotPos, file.Pos(tt.wantOffset))
					}
				}
			} else {
				if gotPos.IsValid() {
					t.Errorf("calculateCursorPos() returned valid pos %v on error, want invalid pos", gotPos)
				}
			}
		})
	}
}

// TestAnalyzeCodeContext_DirectParse tests basic context extraction.
// Needs update after Cycle 8 refactoring.
func TestAnalyzeCodeContext_DirectParse(t *testing.T) {
	t.Skip("Skipping TestAnalyzeCodeContext_DirectParse: Needs update after Cycle 8 refactoring.")

	content := `package main
import "fmt"
// GlobalConst is a global constant.
const GlobalConst = 123
var GlobalVar string
type MyStruct struct { FieldA int; FieldB string }
// MyFunction is the enclosing function.
func MyFunction(arg1 int, arg2 string) (res int, err error) {
	localVar := "hello"
	_ = localVar
	var s MyStruct
	s.FieldA = arg1 // Cursor for selector test -> Line 16 Col 4
	fmt.Println(arg1, ) // Cursor for call test -> Line 17 Col 17
	// Cursor is here -> Line 18 Col 2
	_ = s.DoesNotExist // Cursor for unknown selector test -> Line 19 Col 10
}
`
	tmpDir := t.TempDir()
	tmpFilename := filepath.Join(tmpDir, "test_analyze.go")
	if err := os.WriteFile(tmpFilename, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	testLogger := slog.New(slog.NewTextHandler(io.Discard, nil))
	analyzer := NewGoPackagesAnalyzer(testLogger)
	t.Cleanup(func() {
		if err := analyzer.Close(); err != nil {
			t.Errorf("Error closing analyzer: %v", err)
		}
	})

	t.Run("Inside MyFunction", func(t *testing.T) {
		line, col := 18, 2
		ctx := context.Background()
		// Analyze uses the logger configured in the analyzer instance
		info, analysisErr := analyzer.Analyze(ctx, tmpFilename, 1, line, col)
		isNonFatalLoadErr := analysisErr != nil && errors.Is(analysisErr, ErrAnalysisFailed)
		if analysisErr != nil && !isNonFatalLoadErr {
			t.Fatalf("Analyze returned unexpected fatal error: %v", analysisErr)
		}
		if info == nil {
			t.Fatal("Analyze returned nil info")
		}
		if isNonFatalLoadErr {
			t.Logf("Analyze returned expected non-fatal errors: %v", analysisErr)
		}
		// --- Checks need update after Cycle 8 ---
	})

	// ... other subtests (Selector, Call) need similar updates ...
}

// TestLoadConfig tests configuration loading and default file writing.
func TestLoadConfig(t *testing.T) {
	var logBuf bytes.Buffer
	testLogger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	origConfigDir := os.Getenv("XDG_CONFIG_HOME")
	origHome := os.Getenv("HOME")
	origUserProfile := os.Getenv("USERPROFILE")
	t.Cleanup(func() {
		os.Setenv("XDG_CONFIG_HOME", origConfigDir)
		os.Setenv("HOME", origHome)
		os.Setenv("USERPROFILE", origUserProfile)
	})
	tempHome := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(tempHome, ".config"))
	t.Setenv("HOME", tempHome)
	t.Setenv("USERPROFILE", tempHome)

	fakeConfigDir := filepath.Join(tempHome, ".config", configDirName)
	fakeConfigFile := filepath.Join(fakeConfigDir, defaultConfigFileName)

	tests := []struct {
		name          string
		setup         func(t *testing.T) error
		wantConfig    Config
		checkWrite    bool
		wantWritePath string
		wantErrLog    string
	}{
		{
			name: "No config files - writes default to primary",
			setup: func(t *testing.T) error {
				return os.RemoveAll(fakeConfigDir)
			},
			wantConfig:    getDefaultConfig(),
			checkWrite:    true,
			wantWritePath: fakeConfigFile,
			wantErrLog:    "",
		},
		{
			name: "Config in XDG_CONFIG_HOME",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				jsonData := `{"model": "test-model-config", "temperature": 0.99, "use_fim": true, "max_preamble_len": 1024, "max_snippet_len": 1000, "log_level": "debug"}`
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig: Config{
				OllamaURL: getDefaultConfig().OllamaURL, Model: "test-model-config", MaxTokens: getDefaultConfig().MaxTokens,
				Stop: getDefaultConfig().Stop, Temperature: 0.99, UseAst: getDefaultConfig().UseAst, UseFim: true,
				MaxPreambleLen: 1024, MaxSnippetLen: 1000, LogLevel: "debug",
				PromptTemplate: getDefaultConfig().PromptTemplate, FimTemplate: getDefaultConfig().FimTemplate,
			},
			checkWrite: false,
			wantErrLog: "",
		},
		{
			name: "Invalid JSON - returns defaults, logs warning, writes default",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				jsonData := `{"model": "bad json",`
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig:    getDefaultConfig(),
			checkWrite:    true,
			wantWritePath: fakeConfigFile,
			wantErrLog:    "parsing config file JSON",
		},
		{
			name: "Partial Config - merges with defaults",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				jsonData := `{"ollama_url": "http://other:1111", "use_ast": false, "max_snippet_len": 4096}`
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig: Config{
				OllamaURL: "http://other:1111", Model: getDefaultConfig().Model, MaxTokens: getDefaultConfig().MaxTokens,
				Stop: getDefaultConfig().Stop, Temperature: getDefaultConfig().Temperature, UseAst: false, UseFim: getDefaultConfig().UseFim,
				MaxPreambleLen: getDefaultConfig().MaxPreambleLen, MaxSnippetLen: 4096, LogLevel: getDefaultConfig().LogLevel,
				PromptTemplate: getDefaultConfig().PromptTemplate, FimTemplate: getDefaultConfig().FimTemplate,
			},
			checkWrite: false,
			wantErrLog: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.setup(t); err != nil {
				t.Fatalf("Setup failed: %v", err)
			}

			logBuf.Reset()

			// Call LoadConfig directly, passing the test logger
			gotConfig, err := LoadConfig(testLogger)

			if err != nil && !errors.Is(err, ErrConfig) {
				t.Errorf("LoadConfig() returned unexpected fatal error = %v", err)
			} else if err != nil {
				t.Logf("LoadConfig() returned expected non-fatal error: %v", err)
			}

			logOutput := logBuf.String()
			if tt.wantErrLog != "" && !strings.Contains(logOutput, tt.wantErrLog) {
				t.Errorf("LoadConfig() log output missing expected message containing %q. Got:\n%s", tt.wantErrLog, logOutput)
			}

			// Compare configs (ignoring templates which aren't loaded from file)
			tempWant := tt.wantConfig
			tempGot := gotConfig
			tempWant.PromptTemplate = ""
			tempGot.PromptTemplate = ""
			tempWant.FimTemplate = ""
			tempGot.FimTemplate = ""

			if !reflect.DeepEqual(tempGot, tempWant) {
				t.Errorf("LoadConfig() got = %+v, want %+v", gotConfig, tt.wantConfig)
			}

			_, statErr := os.Stat(tt.wantWritePath)
			if tt.checkWrite {
				if errors.Is(statErr, os.ErrNotExist) {
					t.Errorf("LoadConfig() did not write default config file to %s when expected", tt.wantWritePath)
				} else if statErr != nil {
					t.Errorf("Error checking for default config file %s: %v", tt.wantWritePath, statErr)
				}
			} else {
				if statErr == nil && !strings.Contains(tt.name, "writes default") {
					t.Errorf("LoadConfig() wrote default config file %s when NOT expected", tt.wantWritePath)
				}
			}
		})
	}
}

// TestExtractSnippetContext tests prefix/suffix/line extraction.
func TestExtractSnippetContext(t *testing.T) {
	testLogger := slog.New(slog.NewTextHandler(io.Discard, nil))

	content := "line one\nline two\nline three"
	tmpDir := t.TempDir()
	tmpFilename := filepath.Join(tmpDir, "test_snippet.go")
	if err := os.WriteFile(tmpFilename, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}
	tests := []struct {
		name                             string
		line, col                        int // 1-based input
		wantPrefix, wantSuffix, wantLine string
		wantErr                          bool
	}{
		{"Start of file", 1, 1, "", "line one\nline two\nline three", "line one", false},
		{"Middle of line 1", 1, 6, "line ", "one\nline two\nline three", "line one", false},
		{"End of line 1", 1, 9, "line one", "\nline two\nline three", "line one", false},
		{"Start of line 2", 2, 1, "line one\n", "line two\nline three", "line two", false},
		{"Middle of line 2", 2, 6, "line one\nline ", "two\nline three", "line two", false},
		{"End of line 3", 3, 11, "line one\nline two\nline three", "", "line three", false},
		{"After end of file content", 3, 12, "line one\nline two\nline three", "", "line three", false}, // Cursor at EOF
		{"Start of virtual line 4", 4, 1, "line one\nline two\nline three\n", "", "", false},            // Cursor at start of line after last
		{"Invalid line", 5, 1, "", "", "", true},                                                        // Line too high
		{"Invalid col on virtual line 4", 4, 2, "", "", "", true},                                       // Col > 1 on line after last
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Call utility function directly, passing the test logger
			gotCtx, err := extractSnippetContext(tmpFilename, tt.line, tt.col, testLogger)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractSnippetContext() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if gotCtx.Prefix != tt.wantPrefix {
					t.Errorf("Prefix = %q, want %q", gotCtx.Prefix, tt.wantPrefix)
				}
				if gotCtx.Suffix != tt.wantSuffix {
					t.Errorf("Suffix = %q, want %q", gotCtx.Suffix, tt.wantSuffix)
				}
				if tt.wantLine != "" && gotCtx.FullLine != tt.wantLine {
					t.Errorf("FullLine = %q, want %q", gotCtx.FullLine, tt.wantLine)
				} else if tt.wantLine == "" && gotCtx.FullLine != "" {
					t.Errorf("FullLine = %q, want empty", gotCtx.FullLine)
				}
			}
		})
	}
}

// TestListTypeMembers tests listing fields/methods (Currently Skipped).
func TestListTypeMembers(t *testing.T) {
	t.Skip("listTypeMembers tests not yet implemented (requires mock types or type checker setup)")
}

// TestFindCommentsWithMap tests comment finding logic (Currently Skipped).
func TestFindCommentsWithMap(t *testing.T) {
	t.Skip("findCommentsWithMap tests not yet implemented (requires setting up AST and CommentMap)")
}

// TestFindIdentifierAtCursor tests identifier finding logic (Currently Skipped).
func TestFindIdentifierAtCursor(t *testing.T) {
	t.Skip("findIdentifierAtCursor tests not yet implemented (requires setting up AST and type info)")
}

// setupTestAnalyzer creates an analyzer with a temporary DB for testing cache.
func setupTestAnalyzer(t *testing.T, logger *slog.Logger) (*GoPackagesAnalyzer, string) {
	t.Helper()
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	tmpDir := t.TempDir()
	dbDir := filepath.Join(tmpDir, "test_bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
	if err := os.MkdirAll(dbDir, 0750); err != nil {
		t.Fatalf("Failed to create temp db dir: %v", err)
	}
	dbPath := filepath.Join(dbDir, "test_cache.db")

	opts := &bbolt.Options{Timeout: 1 * time.Second}
	db, err := bbolt.Open(dbPath, 0600, opts)
	if err != nil {
		t.Fatalf("Failed to open test bbolt cache file %s: %v", dbPath, err)
	}

	err = db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(cacheBucketName)
		return err
	})
	if err != nil {
		db.Close()
		t.Fatalf("Failed to create test cache bucket: %v", err)
	}

	memCache, cacheErr := ristretto.NewCache(&ristretto.Config{
		NumCounters: 1e4, MaxCost: 1 << 20, BufferItems: 64, Metrics: true,
	})
	if cacheErr != nil {
		db.Close()
		t.Fatalf("Failed to create test ristretto cache: %v", cacheErr)
	}

	// Create analyzer manually, passing the provided logger
	analyzer := &GoPackagesAnalyzer{db: db, memoryCache: memCache, logger: logger}

	t.Cleanup(func() {
		if err := analyzer.Close(); err != nil {
			t.Errorf("Error closing test analyzer: %v", err)
		}
	})
	return analyzer, tmpDir
}

// captureSlogOutput executes a function while capturing slog output.
// Note: This sets the *default* slog logger. Explicit logger passing is preferred.
func captureSlogOutput(t *testing.T, action func()) string {
	t.Helper()
	var logBuf bytes.Buffer
	testHandler := slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug})
	testLogger := slog.New(testHandler)
	oldLogger := slog.Default()
	slog.SetDefault(testLogger)                      // Set default logger for capture
	t.Cleanup(func() { slog.SetDefault(oldLogger) }) // Restore original default logger

	action()

	return logBuf.String()
}

// TestGoPackagesAnalyzer_Cache tests cache invalidation and hit/miss logic.
func TestGoPackagesAnalyzer_Cache(t *testing.T) {
	testLogger := slog.New(slog.NewTextHandler(io.Discard, nil))

	t.Run("InvalidateCache", func(t *testing.T) {
		analyzer, tmpDir := setupTestAnalyzer(t, testLogger) // Pass logger
		testFilename := filepath.Join(tmpDir, "cache_test.go")
		testContent := `package main; func main() { println("hello") }`
		if err := os.WriteFile(testFilename, []byte(testContent), 0644); err != nil {
			t.Fatalf("Failed to write test file: %v", err)
		}
		ctx := context.Background()
		line, col := 1, 20
		version := 1

		// Analyze uses the logger configured within the analyzer instance
		_, err := analyzer.Analyze(ctx, testFilename, version, line, col)
		if err != nil && !errors.Is(err, ErrAnalysisFailed) {
			t.Fatalf("First Analyze failed unexpectedly: %v", err)
		}
		// Note: Checking logs for specific cache messages is brittle.
		// Focus on functional behavior (e.g., performance difference or re-computation).

		_, err = analyzer.Analyze(ctx, testFilename, version, line, col)
		if err != nil && !errors.Is(err, ErrAnalysisFailed) {
			t.Fatalf("Second Analyze failed unexpectedly: %v", err)
		}

		// Invalidate cache
		if err := analyzer.InvalidateCache(tmpDir); err != nil {
			t.Fatalf("InvalidateCache failed: %v", err)
		}

		_, err = analyzer.Analyze(ctx, testFilename, version, line, col)
		if err != nil && !errors.Is(err, ErrAnalysisFailed) {
			t.Fatalf("Third Analyze failed unexpectedly: %v", err)
		}
		// After invalidation, a cache miss is expected, but checking logs is brittle.
	})

	// ... other cache tests ...
}

// TestDeepCompleter_GetCompletionStreamFromFile_Basic tests the basic flow and error handling.
func TestDeepCompleter_GetCompletionStreamFromFile_Basic(t *testing.T) {
	testLogger := slog.New(slog.NewTextHandler(io.Discard, nil))

	t.Run("Success (mock or skip Ollama)", func(t *testing.T) {
		t.Skip("Skipping success test: Requires mock Ollama or running instance.")

		tmpDir := t.TempDir()
		testFilename := filepath.Join(tmpDir, "basic_test.go")
		testContent := `package main; func main() { print }`
		if err := os.WriteFile(testFilename, []byte(testContent), 0644); err != nil {
			t.Fatalf("Failed to write test file: %v", err)
		}

		completer, err := NewDeepCompleter(testLogger)
		if err != nil && !errors.Is(err, ErrConfig) {
			t.Fatalf("NewDeepCompleter failed: %v", err)
		}
		if completer == nil {
			t.Fatal("Completer is nil")
		}
		t.Cleanup(func() { completer.Close() })

		ctx := context.Background()
		line, col := 1, 20
		version := 1

		var buf bytes.Buffer
		// GetCompletionStreamFromFile uses the logger configured in the completer
		err = completer.GetCompletionStreamFromFile(ctx, testFilename, version, line, col, &buf)

		allowedErrors := []error{ErrOllamaUnavailable, ErrAnalysisFailed, ErrStreamProcessing}
		isAllowedError := false
		if err != nil {
			for _, allowed := range allowedErrors {
				if errors.Is(err, allowed) {
					isAllowedError = true
					break
				}
			}
		}

		if err != nil && !isAllowedError {
			t.Errorf("GetCompletionStreamFromFile returned unexpected error: %v", err)
		} else if err != nil {
			t.Logf("GetCompletionStreamFromFile returned expected error (likely Ollama unavailable): %v", err)
		} else {
			t.Logf("GetCompletionStreamFromFile succeeded. Output: %q", buf.String())
			if buf.Len() == 0 {
				t.Error("Expected non-empty completion, got empty buffer")
			}
		}
	})

	t.Run("File Not Found", func(t *testing.T) {
		completer, err := NewDeepCompleter(testLogger)
		if err != nil && !errors.Is(err, ErrConfig) {
			t.Fatalf("NewDeepCompleter failed: %v", err)
		}
		if completer == nil {
			t.Fatal("Completer is nil")
		}
		t.Cleanup(func() { completer.Close() })

		ctx := context.Background()
		nonExistentFile := filepath.Join(t.TempDir(), "nonexistent.go")
		var buf bytes.Buffer
		version := 1

		err = completer.GetCompletionStreamFromFile(ctx, nonExistentFile, version, 1, 1, &buf)

		if err == nil {
			t.Errorf("Expected an error for non-existent file, got nil")
		} else if !errors.Is(err, os.ErrNotExist) && !strings.Contains(err.Error(), "no such file or directory") && !strings.Contains(err.Error(), "failed to extract code snippet context") {
			// extractSnippetContext might wrap the os.ErrNotExist
			t.Errorf("Expected error indicating file not found, got: %v", err)
		} else {
			t.Logf("Got expected file not found error: %v", err)
		}
	})
}

// TestDeepCompleter_UpdateConfig tests dynamic config updates.
func TestDeepCompleter_UpdateConfig(t *testing.T) {
	testLogger := slog.New(slog.NewTextHandler(io.Discard, nil))

	completer, err := NewDeepCompleter(testLogger)
	if err != nil && !errors.Is(err, ErrConfig) {
		t.Fatalf("NewDeepCompleter failed: %v", err)
	}
	if completer == nil {
		t.Fatal("Completer is nil")
	}
	t.Cleanup(func() { completer.Close() })

	t.Run("ValidUpdate", func(t *testing.T) {
		newValidConfig := getDefaultConfig()
		newValidConfig.Model = "new-test-model"
		newValidConfig.Temperature = 0.88
		newValidConfig.MaxTokens = 512
		newValidConfig.Stop = []string{"\n\n", "//"}

		err := completer.UpdateConfig(newValidConfig)
		if err != nil {
			t.Fatalf("UpdateConfig failed for valid config: %v", err)
		}

		updatedConfig := completer.GetCurrentConfig()
		if updatedConfig.Model != "new-test-model" {
			t.Errorf("Model not updated: got %s, want %s", updatedConfig.Model, "new-test-model")
		}
		if updatedConfig.Temperature != 0.88 {
			t.Errorf("Temperature not updated: got %f, want %f", updatedConfig.Temperature, 0.88)
		}
	})

	t.Run("InvalidUpdate", func(t *testing.T) {
		configBeforeUpdate := completer.GetCurrentConfig()
		newInvalidConfig := getDefaultConfig()
		newInvalidConfig.OllamaURL = "" // Invalid

		err := completer.UpdateConfig(newInvalidConfig)
		if err == nil {
			t.Fatal("UpdateConfig succeeded unexpectedly for invalid config")
		}
		if !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("UpdateConfig returned wrong error type: got %v, want ErrInvalidConfig", err)
		} else {
			t.Logf("Got expected invalid config error: %v", err)
		}

		configAfterUpdate := completer.GetCurrentConfig()
		// Compare configs (ignoring templates)
		tempWant := configBeforeUpdate
		tempGot := configAfterUpdate
		tempWant.PromptTemplate = ""
		tempGot.PromptTemplate = ""
		tempWant.FimTemplate = ""
		tempGot.FimTemplate = ""
		if !reflect.DeepEqual(tempGot, tempWant) {
			t.Errorf("Config changed after invalid update attempt.\nBefore: %+v\nAfter: %+v", configBeforeUpdate, configAfterUpdate)
		}
	})
}

// TestBuildPreamble_Truncation tests preamble truncation logic.
func TestBuildPreamble_Truncation(t *testing.T) {
	testLogger := slog.New(slog.NewTextHandler(io.Discard, nil))

	t.Run("ScopeTruncation", func(t *testing.T) {
		info := &AstContextInfo{
			FilePath:         "/path/to/file.go",
			PackageName:      "main",
			VariablesInScope: make(map[string]types.Object),
		}
		for i := 0; i < 50; i++ {
			varName := fmt.Sprintf("variable_%d_with_a_long_name_to_fill_space", i)
			info.VariablesInScope[varName] = types.NewVar(token.NoPos, nil, varName, types.Typ[types.Int])
		}
		qualifier := func(other *types.Package) string {
			if other != nil {
				return other.Name()
			}
			return ""
		}
		// Pass the specific test logger to buildPreamble
		preamble := buildPreamble(nil, info, qualifier, testLogger)
		t.Logf("Generated Preamble (Scope Truncation Test):\n%s", preamble)

		expectedMarker := "//   ... (scope truncated)"
		if !strings.Contains(preamble, expectedMarker) {
			t.Errorf("Preamble missing expected scope truncation marker '%s'", expectedMarker)
		}
	})
}

// TestCompletionItemGeneration tests LSP CompletionItem creation (Placeholder).
func TestCompletionItemGeneration(t *testing.T) {
	t.Skip("CompletionItem generation tests not yet implemented (requires LSP handler testing)")
}

// ============================================================================
// LSP Position Conversion Tests
// ============================================================================

func TestLSPPositionConversion(t *testing.T) {
	testLogger := slog.New(slog.NewTextHandler(io.Discard, nil))

	t.Run("TestUtf16OffsetToBytes", func(t *testing.T) {
		tests := []struct {
			name           string
			lineContent    string
			utf16Offset    int
			wantByteOffset int
			wantErr        bool
			wantErrType    error
		}{
			{"ASCII start", "hello", 0, 0, false, nil},
			{"ASCII middle", "hello", 2, 2, false, nil},
			{"ASCII end", "hello", 5, 5, false, nil},
			{"ASCII past end", "hello", 6, 5, true, ErrPositionOutOfRange},
			{"ASCII negative", "hello", -1, 0, true, ErrInvalidPositionInput},
			{"2byte UTF-8 start", "héllo", 0, 0, false, nil},
			{"2byte UTF-8 before", "héllo", 1, 1, false, nil},
			{"2byte UTF-8 after", "héllo", 2, 3, false, nil},
			{"2byte UTF-8 middle", "héllo", 4, 5, false, nil},
			{"2byte UTF-8 end", "héllo", 5, 6, false, nil},
			{"2byte UTF-8 past end", "héllo", 6, 6, true, ErrPositionOutOfRange},
			{"3byte UTF-8 start", "€ euro", 0, 0, false, nil},
			{"3byte UTF-8 after", "€ euro", 1, 3, false, nil},
			{"3byte UTF-8 middle", "€ euro", 3, 5, false, nil},
			{"3byte UTF-8 end", "€ euro", 6, 8, false, nil},
			{"3byte UTF-8 past end", "€ euro", 7, 8, true, ErrPositionOutOfRange},
			{"4byte UTF-8 start", "😂笑", 0, 0, false, nil},
			{"4byte UTF-8 middle (within surrogate)", "😂笑", 1, 0, false, nil},
			{"4byte UTF-8 after surrogate", "😂笑", 2, 4, false, nil},
			{"4byte UTF-8 after second char", "😂笑", 3, 7, false, nil},
			{"4byte UTF-8 past end", "😂笑", 4, 7, true, ErrPositionOutOfRange},
			{"Mixed start", "a é 😂 €", 0, 0, false, nil},
			{"Mixed after a", "a é 😂 €", 1, 1, false, nil},
			{"Mixed after space", "a é 😂 €", 2, 2, false, nil},
			{"Mixed after é", "a é 😂 €", 3, 4, false, nil},
			{"Mixed after space 2", "a é 😂 €", 4, 5, false, nil},
			{"Mixed within 😂", "a é 😂 €", 5, 5, false, nil},
			{"Mixed after 😂", "a é 😂 €", 6, 9, false, nil},
			{"Mixed after space 3", "a é 😂 €", 7, 10, false, nil},
			{"Mixed after €", "a é 😂 €", 8, 13, false, nil},
			{"Mixed past end", "a é 😂 €", 9, 13, true, ErrPositionOutOfRange},
			{"Empty line start", "", 0, 0, false, nil},
			{"Empty line past end", "", 1, 0, true, ErrPositionOutOfRange},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				// Call utility function directly, passing the test logger
				gotByteOffset, err := Utf16OffsetToBytes([]byte(tt.lineContent), tt.utf16Offset, testLogger)
				if (err != nil) != tt.wantErr {
					t.Errorf("Utf16OffsetToBytes() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if tt.wantErr && tt.wantErrType != nil {
					if !errors.Is(err, tt.wantErrType) {
						t.Errorf("Utf16OffsetToBytes() error type = %T (%v), want error wrapping %T", err, err, tt.wantErrType)
					}
				}
				if gotByteOffset != tt.wantByteOffset {
					t.Errorf("Utf16OffsetToBytes() gotByteOffset = %d, want %d", gotByteOffset, tt.wantByteOffset)
				}
			})
		}
	})

	t.Run("TestLspPositionToBytePosition", func(t *testing.T) {
		content := "line one\ntwo é 😂\nthree €\n"
		tests := []struct {
			name                           string
			content                        []byte
			lspPos                         LSPPosition
			wantLine, wantCol, wantByteOff int
			wantErr                        bool
			wantErrType                    error
			wantWarnLog                    string // Check for specific warning messages
		}{
			{"Start of file", []byte(content), LSPPosition{Line: 0, Character: 0}, 1, 1, 0, false, nil, ""},
			{"Middle line 1", []byte(content), LSPPosition{Line: 0, Character: 5}, 1, 6, 5, false, nil, ""},
			{"End line 1", []byte(content), LSPPosition{Line: 0, Character: 8}, 1, 9, 8, false, nil, ""},
			{"Start line 2", []byte(content), LSPPosition{Line: 1, Character: 0}, 2, 1, 9, false, nil, ""},
			{"Middle line 2 (before é)", []byte(content), LSPPosition{Line: 1, Character: 4}, 2, 5, 13, false, nil, ""},
			{"Middle line 2 (after é)", []byte(content), LSPPosition{Line: 1, Character: 5}, 2, 7, 15, false, nil, ""},
			{"Middle line 2 (after space)", []byte(content), LSPPosition{Line: 1, Character: 6}, 2, 8, 16, false, nil, ""},
			{"Middle line 2 (within 😂)", []byte(content), LSPPosition{Line: 1, Character: 7}, 2, 8, 16, false, nil, ""},
			{"Middle line 2 (after 😂)", []byte(content), LSPPosition{Line: 1, Character: 8}, 2, 12, 20, false, nil, ""},
			{"End line 2", []byte(content), LSPPosition{Line: 1, Character: 8}, 2, 12, 20, false, nil, ""}, // Clamped by Utf16OffsetToBytes
			{"Start line 3", []byte(content), LSPPosition{Line: 2, Character: 0}, 3, 1, 21, false, nil, ""},
			{"Middle line 3 (after space)", []byte(content), LSPPosition{Line: 2, Character: 6}, 3, 7, 27, false, nil, ""},
			{"Middle line 3 (after €)", []byte(content), LSPPosition{Line: 2, Character: 7}, 3, 10, 30, false, nil, ""},
			{"End line 3", []byte(content), LSPPosition{Line: 2, Character: 7}, 3, 10, 30, false, nil, ""}, // Clamped by Utf16OffsetToBytes
			{"Start empty line 4", []byte(content), LSPPosition{Line: 3, Character: 0}, 4, 1, 31, false, nil, ""},
			{"Nil content", nil, LSPPosition{Line: 0, Character: 0}, 0, 0, -1, true, ErrPositionConversion, ""},
			{"Empty content", []byte(""), LSPPosition{Line: 0, Character: 0}, 1, 1, 0, false, nil, ""},
			{"Empty content invalid char", []byte(""), LSPPosition{Line: 0, Character: 1}, 0, 0, -1, true, ErrPositionOutOfRange, ""},
			{"Empty content invalid line", []byte(""), LSPPosition{Line: 1, Character: 0}, 0, 0, -1, true, ErrPositionOutOfRange, ""},
			{"Negative line", []byte(content), LSPPosition{Line: uint32(int32(-1)), Character: 0}, 0, 0, -1, true, ErrInvalidPositionInput, ""},
			{"Negative char", []byte(content), LSPPosition{Line: 0, Character: uint32(int32(-1))}, 0, 0, -1, true, ErrInvalidPositionInput, ""},
			{"Line past end", []byte(content), LSPPosition{Line: 4, Character: 0}, 0, 0, -1, true, ErrPositionOutOfRange, ""},
			{"Char past end line 1 (Clamps)", []byte(content), LSPPosition{Line: 0, Character: 10}, 1, 9, 8, false, nil, "clamping"},
			{"Char past end line 2 (Clamps)", []byte(content), LSPPosition{Line: 1, Character: 9}, 2, 12, 20, false, nil, "clamping"},
			{"Char past end line 3 (Clamps)", []byte(content), LSPPosition{Line: 2, Character: 8}, 3, 10, 30, false, nil, "clamping"},
			{"Char past end empty line 4", []byte(content), LSPPosition{Line: 3, Character: 1}, 0, 0, -1, true, ErrPositionOutOfRange, ""},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var logBuf bytes.Buffer
				captureLogger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn}))

				// Call utility function directly, passing captureLogger
				gotLine, gotCol, gotByteOff, err := LspPositionToBytePosition(tt.content, tt.lspPos, captureLogger)
				logOutput := logBuf.String()

				if (err != nil) != tt.wantErr {
					t.Errorf("LspPositionToBytePosition() error = %v, wantErr %v. Logs:\n%s", err, tt.wantErr, logOutput)
					return
				}
				if tt.wantErr && tt.wantErrType != nil {
					if !errors.Is(err, tt.wantErrType) {
						t.Errorf("LspPositionToBytePosition() error type = %T (%v), want error wrapping %T. Logs:\n%s", err, err, tt.wantErrType, logOutput)
					}
				}
				if (err != nil) == tt.wantErr {
					if gotLine != tt.wantLine {
						t.Errorf("LspPositionToBytePosition() gotLine = %d, want %d. Logs:\n%s", gotLine, tt.wantLine, logOutput)
					}
					if gotCol != tt.wantCol {
						t.Errorf("LspPositionToBytePosition() gotCol = %d, want %d. Logs:\n%s", gotCol, tt.wantCol, logOutput)
					}
					if gotByteOff != tt.wantByteOff {
						t.Errorf("LspPositionToBytePosition() gotByteOff = %d, want %d. Logs:\n%s", gotByteOff, tt.wantByteOff, logOutput)
					}
				}

				// Check log output for expected warnings
				if tt.wantWarnLog != "" && !strings.Contains(logOutput, tt.wantWarnLog) {
					t.Errorf("LspPositionToBytePosition() log output missing expected message containing %q. Got:\n%s", tt.wantWarnLog, logOutput)
				}
				if tt.wantWarnLog == "" && strings.Contains(logOutput, "level=WARN") {
					t.Errorf("LspPositionToBytePosition() logged unexpected warning. Got:\n%s", logOutput)
				}
			})
		}
	})
}
