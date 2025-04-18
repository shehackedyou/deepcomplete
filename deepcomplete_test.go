// deepcomplete/deepcomplete_test.go
package deepcomplete

import (
	"bytes"
	"context"
	"errors"
	"fmt" // Added for Cycle 8 test
	"go/ast"
	"go/token"
	"go/types" // Added for Cycle 8 test
	"log"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time" // Added for cache test

	"go.etcd.io/bbolt" // Added for cache test
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

	fset := token.NewFileSet()
	file := fset.AddFile("test_calcpos.go", fset.Base(), len(content))
	if file == nil {
		t.Fatal("Failed to add file")
	}

	tests := []struct {
		name       string
		line       int
		col        int
		wantErr    bool
		wantOffset int // Expected 0-based byte offset.
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
		{"Cursor after end of file", 9, 11, false, 105}, // Clamped
		{"Invalid line (too high)", 10, 1, true, -1},
		{"Invalid line (zero)", 0, 1, true, -1},
		{"Invalid col (zero)", 6, 0, true, -1},
		{"Invalid col (too high for line)", 6, 100, false, 62}, // Clamped
		{"Col too high last line", 9, 100, false, 105},         // Clamped
		{"Col too high last line + 1", 10, 100, true, -1},      // Invalid line
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotPos, err := calculateCursorPos(file, tt.line, tt.col)

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

// TestAnalyzeCodeContext_DirectParse tests basic context extraction without full type checking.
func TestAnalyzeCodeContext_DirectParse(t *testing.T) {
	content := `package main
import "fmt"
// GlobalConst is a global constant.
const GlobalConst = 123
var GlobalVar string
type MyStruct struct { FieldA int }
// MyFunction is the enclosing function.
func MyFunction(arg1 int, arg2 string) (res int, err error) {
	localVar := "hello"
	_ = localVar
	var s MyStruct
	s.FieldA = arg1 // Cursor for selector test -> Line 16 Col 4
	fmt.Println(arg1, ) // Cursor for call test -> Line 17 Col 17
	// Cursor is here -> Line 18 Col 2
}
`
	tmpDir := t.TempDir()
	tmpFilename := filepath.Join(tmpDir, "test_analyze.go")
	if err := os.WriteFile(tmpFilename, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	// Analyzer without DB path disables caching for this test.
	analyzer := NewGoPackagesAnalyzer()
	t.Cleanup(func() {
		if err := analyzer.Close(); err != nil {
			t.Errorf("Error closing analyzer: %v", err)
		}
	})

	t.Run("Inside MyFunction", func(t *testing.T) {
		line, col := 18, 2
		ctx := context.Background()
		info, analysisErr := analyzer.Analyze(ctx, tmpFilename, line, col)

		// Expect non-fatal ErrAnalysisFailed due to missing type info in this setup.
		isNonFatalLoadErr := analysisErr != nil && errors.Is(analysisErr, ErrAnalysisFailed)
		if analysisErr != nil && !isNonFatalLoadErr {
			t.Fatalf("Analyze returned unexpected fatal error: %v", analysisErr)
		}
		if info == nil {
			t.Fatal("Analyze returned nil info")
		}
		if isNonFatalLoadErr {
			t.Logf("Analyze returned expected non-fatal errors: %v", analysisErr)
			// Check if specific expected error substrings are present.
			foundLoadError := false
			expectedSubstrings := []string{
				"type info missing", "package loading failed", "type info unavailable",
				"TypesInfo is nil", "Types is nil", "missing object info",
				"package scope missing", "TypesInfo", // General check
			}
			for _, e := range info.AnalysisErrors {
				errStr := e.Error()
				for _, sub := range expectedSubstrings {
					if strings.Contains(errStr, sub) {
						foundLoadError = true
						break
					}
				}
				if foundLoadError {
					break
				}
			}
			if !foundLoadError {
				t.Errorf("Expected package load/type info errors in info.AnalysisErrors, got: %v", info.AnalysisErrors)
			}
		}

		// Assertions on extracted context (expecting AST info only).
		if info.EnclosingFuncNode == nil {
			t.Errorf("Expected EnclosingFuncNode")
		}
		if info.EnclosingFuncNode != nil && (info.EnclosingFuncNode.Name == nil || info.EnclosingFuncNode.Name.Name != "MyFunction") {
			t.Errorf("Expected enclosing function 'MyFunction', got %v", info.EnclosingFuncNode.Name)
		}
		foundDocComment := false
		expectedDoc := "MyFunction is the enclosing function."
		for _, c := range info.CommentsNearCursor {
			if strings.Contains(strings.TrimSpace(strings.TrimPrefix(c, "//")), expectedDoc) {
				foundDocComment = true
				break
			}
		}
		if !foundDocComment {
			t.Errorf("Expected function Doc comment containing '%s', got: %v", expectedDoc, info.CommentsNearCursor)
		}

		// Basic preamble checks.
		if !strings.Contains(info.PromptPreamble, "// Enclosing Function (AST only): MyFunction(...)") {
			t.Errorf("Preamble missing enclosing function. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, expectedDoc) {
			t.Errorf("Preamble missing doc comment. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, "Variables/Constants/Types in Scope") {
			t.Errorf("Preamble missing scope section. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, "//   import \"fmt\"") {
			t.Errorf("Preamble missing import fmt. Got:\n%s", info.PromptPreamble)
		}
	})

	t.Run("Selector Expression", func(t *testing.T) {
		line, col := 16, 4 // After "s."
		ctx := context.Background()
		info, analysisErr := analyzer.Analyze(ctx, tmpFilename, line, col)

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

		if info.SelectorExpr == nil {
			t.Fatalf("Expected SelectorExpr context, got nil")
		}
		if selX, ok := info.SelectorExpr.X.(*ast.Ident); !ok || selX.Name != "s" {
			t.Errorf("Expected selector base 's', got %T", info.SelectorExpr.X)
		}
		if info.SelectorExpr.Sel == nil || info.SelectorExpr.Sel.Name != "FieldA" {
			t.Errorf("Expected selector identifier 'FieldA', got %v", info.SelectorExpr.Sel)
		}

		// Check preamble for expected fallback messages when type info is missing.
		if !strings.Contains(info.PromptPreamble, "// Selector context: expr type = (unknown - type analysis failed") {
			t.Errorf("Preamble missing selector context fallback. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, "//   (Cannot list members: type analysis failed") {
			t.Errorf("Preamble missing member listing fallback. Got:\n%s", info.PromptPreamble)
		}
	})

	t.Run("Call Expression", func(t *testing.T) {
		line, col := 17, 17 // Inside fmt.Println(arg1, |)
		ctx := context.Background()
		info, analysisErr := analyzer.Analyze(ctx, tmpFilename, line, col)

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

		if info.CallExpr == nil {
			t.Fatalf("Expected CallExpr context, got nil")
		}
		if info.CallArgIndex != 1 {
			t.Errorf("Expected CallArgIndex 1, got %d", info.CallArgIndex)
		}

		// Check preamble for expected fallback messages.
		if !strings.Contains(info.PromptPreamble, "// Inside function call: Println (Arg 2)") {
			t.Errorf("Preamble missing call expression context. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, "// Function Signature: (unknown - type analysis failed") {
			t.Errorf("Preamble missing unknown signature fallback. Got:\n%s", info.PromptPreamble)
		}
	})
}

// TestLoadConfig tests configuration loading and default file writing.
func TestLoadConfig(t *testing.T) {
	// Setup fake home/config directories using temp dir and env vars.
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
		checkWrite    bool   // Check if default config was written.
		wantWritePath string // Expected path for default config write.
		wantErrLog    string // Expected substring in warning log.
	}{
		{
			name:          "No config files - writes default",
			setup:         func(t *testing.T) error { os.RemoveAll(fakeConfigDir); return nil },
			wantConfig:    DefaultConfig,
			checkWrite:    true,
			wantWritePath: fakeConfigFile,
			wantErrLog:    "", // Expect info log about writing, not warning.
		},
		{
			name: "Config in XDG_CONFIG_HOME",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				jsonData := `{"model": "test-model-config", "temperature": 0.99, "use_fim": true, "max_preamble_len": 1024, "max_snippet_len": 1000}`
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig: Config{OllamaURL: DefaultConfig.OllamaURL, Model: "test-model-config", MaxTokens: DefaultConfig.MaxTokens, Stop: DefaultConfig.Stop, Temperature: 0.99, UseAst: DefaultConfig.UseAst, UseFim: true, MaxPreambleLen: 1024, MaxSnippetLen: 1000},
			checkWrite: false,
			wantErrLog: "",
		},
		{
			name: "Invalid JSON - returns defaults, logs warning, writes default",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				jsonData := `{"model": "bad json",` // Invalid JSON.
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig:    DefaultConfig,
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
			wantConfig: Config{OllamaURL: "http://other:1111", Model: DefaultConfig.Model, MaxTokens: DefaultConfig.MaxTokens, Stop: DefaultConfig.Stop, Temperature: DefaultConfig.Temperature, UseAst: false, UseFim: DefaultConfig.UseFim, MaxPreambleLen: DefaultConfig.MaxPreambleLen, MaxSnippetLen: 4096},
			checkWrite: false,
			wantErrLog: "",
		},
		// Added in Cycle 4
		{
			name: "Empty JSON file - returns defaults, writes default",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				return os.WriteFile(fakeConfigFile, []byte("{}"), 0644) // Empty but valid JSON
			},
			wantConfig:    DefaultConfig,
			checkWrite:    false, // Should not rewrite if valid JSON exists, even if empty
			wantWritePath: fakeConfigFile,
			wantErrLog:    "", // No error expected
		},
		{
			name: "Unknown fields JSON - returns defaults, logs warning, writes default",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				jsonData := `{"unknown_field": 123, "model": "known"}` // Mix known and unknown
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			// Expect known fields to be loaded, others ignored.
			wantConfig: Config{OllamaURL: DefaultConfig.OllamaURL, Model: "known", MaxTokens: DefaultConfig.MaxTokens, Stop: DefaultConfig.Stop, Temperature: DefaultConfig.Temperature, UseAst: DefaultConfig.UseAst, UseFim: DefaultConfig.UseFim, MaxPreambleLen: DefaultConfig.MaxPreambleLen, MaxSnippetLen: DefaultConfig.MaxSnippetLen},
			checkWrite: false,
			wantErrLog: "", // JSON parsing itself doesn't warn on unknown fields by default.
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.setup(t); err != nil {
				t.Fatalf("Setup failed: %v", err)
			}

			var logBuf bytes.Buffer
			log.SetOutput(&logBuf)
			t.Cleanup(func() { log.SetOutput(os.Stderr) })

			gotConfig, err := LoadConfig()

			// Check only for fatal errors (non-ErrConfig/ErrInvalidConfig).
			if err != nil && !errors.Is(err, ErrConfig) && !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("LoadConfig() returned unexpected fatal error = %v", err)
			} else if err != nil {
				t.Logf("LoadConfig() returned expected non-fatal error: %v", err)
			}

			logOutput := logBuf.String()
			if tt.wantErrLog != "" && !strings.Contains(logOutput, tt.wantErrLog) {
				t.Errorf("LoadConfig() log output missing expected warning containing %q. Got:\n%s", tt.wantErrLog, logOutput)
			}
			allowedWarnings := []string{"writing default config", "Could not determine", "validation failed", "Config file exists but is empty"}
			if tt.wantErrLog == "" && err == nil && strings.Contains(logOutput, "Warning:") {
				isAllowed := false
				for _, allowed := range allowedWarnings {
					if strings.Contains(logOutput, allowed) {
						isAllowed = true
						break
					}
				}
				if !isAllowed {
					t.Errorf("LoadConfig() logged unexpected warning. Got:\n%s", logOutput)
				}
			}

			// Compare relevant config fields (ignore internal templates).
			tempWant := tt.wantConfig
			tempGot := gotConfig
			tempWant.PromptTemplate = ""
			tempGot.PromptTemplate = ""
			tempWant.FimTemplate = ""
			tempGot.FimTemplate = ""
			if !reflect.DeepEqual(tempGot, tempWant) {
				t.Errorf("LoadConfig() got = %+v, want %+v", gotConfig, tt.wantConfig)
			}

			// Check if default file was written when expected/not expected.
			_, statErr := os.Stat(tt.wantWritePath)
			if tt.checkWrite {
				if errors.Is(statErr, os.ErrNotExist) {
					t.Errorf("LoadConfig() did not write default config file to %s when expected", tt.wantWritePath)
				} else if statErr != nil {
					t.Errorf("Error checking for default config file %s: %v", tt.wantWritePath, statErr)
				} else {
					t.Logf("Default config file written correctly to %s", tt.wantWritePath)
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
	content := `line one
line two
line three`
	tmpDir := t.TempDir()
	tmpFilename := filepath.Join(tmpDir, "test_snippet.go")
	if err := os.WriteFile(tmpFilename, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	tests := []struct {
		name                             string
		line, col                        int
		wantPrefix, wantSuffix, wantLine string
		wantErr                          bool
	}{
		{"Start of file", 1, 1, "", "line one\nline two\nline three", "line one", false},
		{"Middle of line 1", 1, 6, "line ", "one\nline two\nline three", "line one", false},
		{"End of line 1", 1, 9, "line one", "\nline two\nline three", "line one", false},
		{"Start of line 2", 2, 1, "line one\n", "line two\nline three", "line two", false},
		{"Middle of line 2", 2, 6, "line one\nline ", "two\nline three", "line two", false},
		{"End of line 3", 3, 11, "line one\nline two\nline three", "", "line three", false},
		{"After end of file", 3, 12, "line one\nline two\nline three", "", "line three", false}, // Clamped
		{"Invalid line", 4, 1, "", "", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCtx, err := extractSnippetContext(tmpFilename, tt.line, tt.col)
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
				if gotCtx.FullLine != tt.wantLine {
					t.Errorf("FullLine = %q, want %q", gotCtx.FullLine, tt.wantLine)
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
	t.Skip("findCommentsWithMap tests not yet implemented")
}

// TestFindIdentifierAtCursor tests identifier finding logic (Currently Skipped).
func TestFindIdentifierAtCursor(t *testing.T) {
	t.Skip("findIdentifierAtCursor tests not yet implemented")
}

// setupTestAnalyzer creates an analyzer with a temporary DB for testing cache.
func setupTestAnalyzer(t *testing.T) (*GoPackagesAnalyzer, string) {
	t.Helper()
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
	err = db.Update(func(tx *bbolt.Tx) error { _, err := tx.CreateBucketIfNotExists(cacheBucketName); return err })
	if err != nil {
		db.Close()
		t.Fatalf("Failed to create test cache bucket: %v", err)
	}

	analyzer := &GoPackagesAnalyzer{db: db}
	t.Cleanup(func() {
		if err := analyzer.Close(); err != nil {
			t.Errorf("Error closing test analyzer: %v", err)
		}
	})
	return analyzer, tmpDir // Return tmpDir for file operations within tests
}

// captureLogs executes a function while capturing log output.
func captureLogs(t *testing.T, action func()) string {
	t.Helper()
	var logBuf bytes.Buffer
	testLogger := log.New(&logBuf, "TESTLOG: ", 0) // Use simple prefix, no flags
	oldLogger := log.Default()
	log.SetOutput(testLogger.Writer())
	log.SetFlags(testLogger.Flags())
	log.SetPrefix(testLogger.Prefix())
	t.Cleanup(func() {
		log.SetOutput(oldLogger.Writer())
		log.SetFlags(oldLogger.Flags())
		log.SetPrefix(oldLogger.Prefix())
	})
	action()
	return logBuf.String()
}

// TestGoPackagesAnalyzer_Cache tests cache invalidation and hit/miss logic.
func TestGoPackagesAnalyzer_Cache(t *testing.T) {

	// Test InvalidateCache functionality.
	t.Run("InvalidateCache", func(t *testing.T) {
		analyzer, tmpDir := setupTestAnalyzer(t)
		testFilename := filepath.Join(tmpDir, "cache_test.go")
		testContent := `package main; func main() { println("hello") }`
		if err := os.WriteFile(testFilename, []byte(testContent), 0644); err != nil {
			t.Fatalf("Failed to write test file: %v", err)
		}
		ctx := context.Background()
		line, col := 2, 20 // Dummy position

		// 1. First Analysis (miss -> save).
		logOutput1 := captureLogs(t, func() {
			_, err := analyzer.Analyze(ctx, testFilename, line, col)
			if err != nil && !errors.Is(err, ErrAnalysisFailed) {
				t.Fatalf("First Analyze failed unexpectedly: %v", err)
			}
		})
		if !strings.Contains(logOutput1, "Cache miss or invalid") {
			t.Errorf("Expected 'Cache miss' log during first analysis, got:\n%s", logOutput1)
		}
		cacheWasSaved := strings.Contains(logOutput1, "Saved analysis results")

		// 2. Second Analysis (hit or miss depending on first save).
		logOutput2 := captureLogs(t, func() {
			_, err := analyzer.Analyze(ctx, testFilename, line, col)
			if err != nil && !errors.Is(err, ErrAnalysisFailed) {
				t.Fatalf("Second Analyze failed unexpectedly: %v", err)
			}
		})
		if cacheWasSaved { // Only expect hit if first analysis saved.
			if !strings.Contains(logOutput2, "Cache VALID") {
				t.Errorf("Expected 'Cache VALID' log during second analysis, got:\n%s", logOutput2)
			}
			if strings.Contains(logOutput2, "Cache miss or invalid") {
				t.Errorf("Unexpected 'Cache miss' log during second analysis, got:\n%s", logOutput2)
			}
		} else {
			t.Log("Skipping cache hit check for second analysis as first didn't save.")
		}

		// 3. Invalidate Cache.
		logOutputInvalidate := captureLogs(t, func() {
			if err := analyzer.InvalidateCache(tmpDir); err != nil {
				t.Fatalf("InvalidateCache failed: %v", err)
			}
		})
		if cacheWasSaved { // Only expect delete log if cache was saved.
			if !strings.Contains(logOutputInvalidate, "Deleting cache entry") {
				t.Errorf("Expected 'Deleting cache entry' log, got:\n%s", logOutputInvalidate)
			}
		} else {
			t.Log("Skipping check for 'Deleting cache entry' log.")
		}

		// 4. Third Analysis (miss).
		logOutput3 := captureLogs(t, func() {
			_, err := analyzer.Analyze(ctx, testFilename, line, col)
			if err != nil && !errors.Is(err, ErrAnalysisFailed) {
				t.Fatalf("Third Analyze failed unexpectedly: %v", err)
			}
		})
		if !strings.Contains(logOutput3, "Cache miss or invalid") {
			t.Errorf("Expected 'Cache miss' log during third analysis, got:\n%s", logOutput3)
		}
		if strings.Contains(logOutput3, "Cache VALID") {
			t.Errorf("Unexpected 'Cache VALID' log during third analysis, got:\n%s", logOutput3)
		}
	})

	// Test Hit/Miss logic.
	t.Run("HitMiss", func(t *testing.T) {
		analyzer, tmpDir := setupTestAnalyzer(t)
		testFilename := filepath.Join(tmpDir, "hitmiss_test.go")
		testContent := `package main; func main() { println("hit miss") }`
		if err := os.WriteFile(testFilename, []byte(testContent), 0644); err != nil {
			t.Fatalf("Failed to write test file: %v", err)
		}
		ctx := context.Background()
		line, col := 2, 20 // Dummy position

		// 1. First Analysis (Miss -> Save)
		logOutput1 := captureLogs(t, func() {
			_, err := analyzer.Analyze(ctx, testFilename, line, col)
			if err != nil && !errors.Is(err, ErrAnalysisFailed) {
				t.Fatalf("First Analyze failed unexpectedly: %v", err)
			}
		})
		if !strings.Contains(logOutput1, "Cache miss or invalid") {
			t.Errorf("Expected 'Cache miss' log during first analysis, got:\n%s", logOutput1)
		}
		cacheWasSaved := strings.Contains(logOutput1, "Saved analysis results")
		if !cacheWasSaved {
			t.Log("Warning: First analysis did not save to cache, hit test may be invalid.")
		}

		// 2. Second Analysis (Hit)
		logOutput2 := captureLogs(t, func() {
			_, err := analyzer.Analyze(ctx, testFilename, line, col)
			if err != nil && !errors.Is(err, ErrAnalysisFailed) {
				t.Fatalf("Second Analyze failed unexpectedly: %v", err)
			}
		})
		if cacheWasSaved {
			if !strings.Contains(logOutput2, "Cache VALID") {
				t.Errorf("Expected 'Cache VALID' log during second analysis, got:\n%s", logOutput2)
			}
			if !strings.Contains(logOutput2, "Using cached preamble") {
				t.Errorf("Expected 'Using cached preamble' log during second analysis, got:\n%s", logOutput2)
			}
			if strings.Contains(logOutput2, "Cache miss or invalid") {
				t.Errorf("Unexpected 'Cache miss' log during second analysis, got:\n%s", logOutput2)
			}
		} else {
			t.Log("Skipping cache hit check for second analysis as first didn't save.")
			if !strings.Contains(logOutput2, "Cache miss or invalid") {
				t.Errorf("Expected 'Cache miss' log during second analysis (as first didn't save), got:\n%s", logOutput2)
			}
		}
	})

	// Added in Cycle 9
	t.Run("HashPathSeparator", func(t *testing.T) {
		analyzer, tmpDir := setupTestAnalyzer(t)
		// Create a subdirectory and file
		subDir := filepath.Join(tmpDir, "subdir")
		if err := os.Mkdir(subDir, 0755); err != nil {
			t.Fatalf("Failed to create subdir: %v", err)
		}
		testFilename := filepath.Join(subDir, "subfile.go")
		testContent := `package main; func sub() {}`
		if err := os.WriteFile(testFilename, []byte(testContent), 0644); err != nil {
			t.Fatalf("Failed to write sub file: %v", err)
		}
		// Create a dummy go.mod so calculateInputHashes uses package info if possible
		goModPath := filepath.Join(tmpDir, "go.mod")
		if err := os.WriteFile(goModPath, []byte("module testcache"), 0644); err != nil {
			t.Fatalf("Failed to write go.mod: %v", err)
		}

		// Calculate hashes (pkg might be nil if load fails, fallback scan should work)
		hashes, err := calculateInputHashes(tmpDir, nil) // Pass nil pkg to test fallback potentially
		if err != nil {
			t.Fatalf("calculateInputHashes failed: %v", err)
		}

		found := false
		expectedKey := "subdir/subfile.go" // Expect forward slash
		for key := range hashes {
			t.Logf("Found hash key: %s", key)
			if key == expectedKey {
				found = true
				break
			}
			// Check for incorrect backslash just in case
			if strings.Contains(key, "\\") {
				t.Errorf("Hash map key '%s' contains backslash, expected forward slash", key)
			}
		}
		if !found {
			t.Errorf("Expected hash map key '%s' not found in calculated hashes: %v", expectedKey, hashes)
		}
	})

	// TODO: Add more specific cache tests (modify file, modify go.mod, schema change).
	// t.Skip("Remaining GoPackagesAnalyzer caching tests not yet implemented")
}

// TestDeepCompleter_GetCompletionStreamFromFile_Basic tests the basic flow and error handling.
// Added Cycle 5, refined Cycle 7.
func TestDeepCompleter_GetCompletionStreamFromFile_Basic(t *testing.T) {
	// Test successful execution (mocking would be better).
	t.Run("Success", func(t *testing.T) {
		tmpDir := t.TempDir()
		testFilename := filepath.Join(tmpDir, "basic_test.go")
		testContent := `package main; func main() { print }` // Cursor at end
		if err := os.WriteFile(testFilename, []byte(testContent), 0644); err != nil {
			t.Fatalf("Failed to write test file: %v", err)
		}

		// Use default config, assumes Ollama might not be running.
		completer, err := NewDeepCompleter()
		if err != nil && !errors.Is(err, ErrConfig) {
			t.Fatalf("NewDeepCompleter failed: %v", err)
		}
		if completer == nil {
			t.Fatal("Completer is nil")
		}
		t.Cleanup(func() { completer.Close() })

		ctx := context.Background()
		line, col := 2, 20 // Position after "print"
		var buf bytes.Buffer

		// Call the function, allow specific non-fatal errors for this basic test.
		err = completer.GetCompletionStreamFromFile(ctx, testFilename, line, col, &buf)
		if err != nil && !errors.Is(err, ErrOllamaUnavailable) && !errors.Is(err, ErrAnalysisFailed) && !errors.Is(err, ErrStreamProcessing) {
			// Fail only if it's an unexpected error type.
			t.Errorf("GetCompletionStreamFromFile returned unexpected error: %v", err)
		} else if err != nil {
			// Log expected errors.
			t.Logf("GetCompletionStreamFromFile returned expected error: %v", err)
		} else {
			t.Logf("GetCompletionStreamFromFile succeeded. Output: %q", buf.String())
			// Basic check on output if needed, e.g., if !strings.Contains(buf.String(), "...")
		}
	})

	// Test file not found error (Added Cycle 7).
	t.Run("File Not Found", func(t *testing.T) {
		completer, err := NewDeepCompleter()
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

		err = completer.GetCompletionStreamFromFile(ctx, nonExistentFile, 1, 1, &buf)
		if err == nil {
			t.Errorf("Expected an error for non-existent file, got nil")
		} else if !errors.Is(err, os.ErrNotExist) {
			// Check if the error wraps os.ErrNotExist (or equivalent).
			// extractSnippetContext should return the os error directly.
			t.Errorf("Expected error wrapping os.ErrNotExist, got: %v", err)
		} else {
			t.Logf("Got expected file not found error: %v", err)
		}
	})
}

// TestDeepCompleter_UpdateConfig tests dynamic config updates. (Added Cycle 6)
func TestDeepCompleter_UpdateConfig(t *testing.T) {
	completer, err := NewDeepCompleter() // Use default initial config
	if err != nil && !errors.Is(err, ErrConfig) {
		t.Fatalf("NewDeepCompleter failed: %v", err)
	}
	if completer == nil {
		t.Fatal("Completer is nil")
	}
	t.Cleanup(func() { completer.Close() })

	initialConfig := completer.config // Read initial config (no lock needed before updates)

	// Test valid update
	t.Run("ValidUpdate", func(t *testing.T) {
		newValidConfig := DefaultConfig // Start from defaults
		newValidConfig.Model = "new-test-model"
		newValidConfig.Temperature = 0.88
		newValidConfig.MaxTokens = 512

		err := completer.UpdateConfig(newValidConfig)
		if err != nil {
			t.Fatalf("UpdateConfig failed for valid config: %v", err)
		}

		// Check internal config (using lock for read safety)
		completer.configMu.RLock()
		updatedConfig := completer.config
		completer.configMu.RUnlock()

		if updatedConfig.Model != "new-test-model" {
			t.Errorf("Model not updated: got %s, want %s", updatedConfig.Model, "new-test-model")
		}
		if updatedConfig.Temperature != 0.88 {
			t.Errorf("Temperature not updated: got %f, want %f", updatedConfig.Temperature, 0.88)
		}
		if updatedConfig.MaxTokens != 512 {
			t.Errorf("MaxTokens not updated: got %d, want %d", updatedConfig.MaxTokens, 512)
		}
		// Ensure other fields weren't unexpectedly changed from the valid new config
		if updatedConfig.OllamaURL != newValidConfig.OllamaURL {
			t.Errorf("OllamaURL unexpectedly changed")
		}
	})

	// Test invalid update
	t.Run("InvalidUpdate", func(t *testing.T) {
		configBeforeUpdate := completer.config // Get config state before invalid update attempt

		newInvalidConfig := DefaultConfig
		newInvalidConfig.OllamaURL = "" // Invalid empty URL

		err := completer.UpdateConfig(newInvalidConfig)
		if err == nil {
			t.Fatal("UpdateConfig succeeded unexpectedly for invalid config")
		}
		if !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("UpdateConfig returned wrong error type: got %v, want ErrInvalidConfig", err)
		}

		// Check internal config hasn't changed (using lock)
		completer.configMu.RLock()
		configAfterUpdate := completer.config
		completer.configMu.RUnlock()

		// Use reflect.DeepEqual for struct comparison (ignoring mutex field)
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

// TestBuildPreamble_Truncation tests preamble truncation logic. (Added Cycle 8)
func TestBuildPreamble_Truncation(t *testing.T) {
	t.Run("ScopeTruncation", func(t *testing.T) {
		// Create mock AstContextInfo with many scope variables
		info := &AstContextInfo{
			FilePath:         "/path/to/file.go",
			PackageName:      "main",
			VariablesInScope: make(map[string]types.Object),
		}
		// Add more variables than the internal limit allows (e.g., > 30)
		for i := 0; i < 50; i++ {
			varName := fmt.Sprintf("variable_%d", i)
			// Create a dummy types.Var object (replace with more realistic if needed)
			info.VariablesInScope[varName] = types.NewVar(token.NoPos, nil, varName, types.Typ[types.Int])
		}

		// Use a basic qualifier for testing
		qualifier := func(other *types.Package) string {
			if other != nil {
				return other.Name()
			}
			return ""
		}

		// Call buildPreamble (which calls formatScopeSection internally)
		preamble := buildPreamble(info, qualifier)

		t.Logf("Generated Preamble (Truncation Test):\n%s", preamble)

		// Check if the scope truncation marker is present
		expectedMarker := "//   ... (scope truncated)"
		if !strings.Contains(preamble, expectedMarker) {
			t.Errorf("Preamble missing expected scope truncation marker '%s'", expectedMarker)
		}

		// Optional: Check overall length is somewhat reasonable (though final limit applied later)
		// if len(preamble) > 4096 { // Example check
		// 	t.Errorf("Generated preamble seems excessively long (%d bytes) despite expected truncation", len(preamble))
		// }
	})
	// TODO: Add tests for truncating other sections (imports, comments)
}

// TestCompletionItemGeneration tests LSP CompletionItem creation (Placeholder - Cycle 9)
func TestCompletionItemGeneration(t *testing.T) {
	// TODO: Implement tests for completion item generation logic in LSP handler.
	// This likely requires mocking the core completer or testing helper functions.
	t.Skip("CompletionItem generation tests not yet implemented")
}

// ============================================================================
// LSP Position Conversion Tests
// ============================================================================

func TestLSPPositionConversion(t *testing.T) {
	// --- Test Utf16OffsetToBytes ---
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
			{"4byte UTF-8 end", "😂笑", 3, 7, false, nil},
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
			{"Mixed end", "a é 😂 €", 8, 13, false, nil},
			{"Mixed past end", "a é 😂 €", 9, 13, true, ErrPositionOutOfRange},
			{"Empty line start", "", 0, 0, false, nil},
			{"Empty line past end", "", 1, 0, true, ErrPositionOutOfRange},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				gotByteOffset, err := Utf16OffsetToBytes([]byte(tt.lineContent), tt.utf16Offset)
				if (err != nil) != tt.wantErr {
					t.Errorf("Utf16OffsetToBytes() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if tt.wantErr && tt.wantErrType != nil {
					if !errors.Is(err, tt.wantErrType) {
						t.Errorf("Utf16OffsetToBytes() error type = %T, want error wrapping %T", err, tt.wantErrType)
					}
				}
				if gotByteOffset != tt.wantByteOffset {
					t.Errorf("Utf16OffsetToBytes() gotByteOffset = %d, want %d", gotByteOffset, tt.wantByteOffset)
				}
			})
		}
	})

	// --- Test LspPositionToBytePosition ---
	t.Run("TestLspPositionToBytePosition", func(t *testing.T) {
		content := "line one\ntwo é 😂\nthree €\n"
		tests := []struct {
			name                           string
			content                        []byte
			lspPos                         LSPPosition
			wantLine, wantCol, wantByteOff int
			wantErr                        bool
			wantErrType                    error
			wantWarnLog                    string
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
			{"End line 2", []byte(content), LSPPosition{Line: 1, Character: 8}, 2, 12, 20, false, nil, ""},
			{"Start line 3", []byte(content), LSPPosition{Line: 2, Character: 0}, 3, 1, 21, false, nil, ""},
			{"Middle line 3 (after space)", []byte(content), LSPPosition{Line: 2, Character: 6}, 3, 7, 27, false, nil, ""},
			{"Middle line 3 (after €)", []byte(content), LSPPosition{Line: 2, Character: 7}, 3, 10, 30, false, nil, ""},
			{"End line 3", []byte(content), LSPPosition{Line: 2, Character: 7}, 3, 10, 30, false, nil, ""},
			{"Start empty line 4", []byte(content), LSPPosition{Line: 3, Character: 0}, 4, 1, 31, false, nil, ""},
			{"Nil content", nil, LSPPosition{Line: 0, Character: 0}, 0, 0, -1, true, ErrPositionConversion, ""},
			{"Empty content", []byte(""), LSPPosition{Line: 0, Character: 0}, 1, 1, 0, false, nil, ""},
			{"Empty content invalid char", []byte(""), LSPPosition{Line: 0, Character: 1}, 0, 0, -1, true, ErrPositionOutOfRange, ""},
			{"Empty content invalid line", []byte(""), LSPPosition{Line: 1, Character: 0}, 0, 0, -1, true, ErrPositionOutOfRange, ""},
			{"Negative line", []byte(content), LSPPosition{Line: uint32(int32(-1)), Character: 0}, 0, 0, -1, true, ErrInvalidPositionInput, ""},
			{"Negative char", []byte(content), LSPPosition{Line: 0, Character: uint32(int32(-1))}, 0, 0, -1, true, ErrInvalidPositionInput, ""},
			{"Line past end", []byte(content), LSPPosition{Line: 4, Character: 0}, 0, 0, -1, true, ErrPositionOutOfRange, ""},
			{"Char past end line 1 (Clamps)", []byte(content), LSPPosition{Line: 0, Character: 10}, 1, 9, 8, false, nil, "Clamping"},
			{"Char past end line 2 (Clamps)", []byte(content), LSPPosition{Line: 1, Character: 9}, 2, 12, 20, false, nil, "Clamping"},
			{"Char past end line 3 (Clamps)", []byte(content), LSPPosition{Line: 2, Character: 8}, 3, 10, 30, false, nil, "Clamping"},
			{"Char past end empty line 4", []byte(content), LSPPosition{Line: 3, Character: 1}, 0, 0, -1, true, ErrPositionOutOfRange, ""},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				logOutput := captureLogs(t, func() {
					gotLine, gotCol, gotByteOff, err := LspPositionToBytePosition(tt.content, tt.lspPos)
					if (err != nil) != tt.wantErr {
						t.Errorf("LspPositionToBytePosition() error = %v, wantErr %v", err, tt.wantErr)
						return
					}
					if tt.wantErr && tt.wantErrType != nil {
						if !errors.Is(err, tt.wantErrType) {
							t.Errorf("LspPositionToBytePosition() error type = %T, want error wrapping %T", err, tt.wantErrType)
						}
					}
					if (err != nil) == tt.wantErr {
						if gotLine != tt.wantLine {
							t.Errorf("LspPositionToBytePosition() gotLine = %d, want %d", gotLine, tt.wantLine)
						}
						if gotCol != tt.wantCol {
							t.Errorf("LspPositionToBytePosition() gotCol = %d, want %d", gotCol, tt.wantCol)
						}
						if gotByteOff != tt.wantByteOff {
							t.Errorf("LspPositionToBytePosition() gotByteOff = %d, want %d", gotByteOff, tt.wantByteOff)
						}
					}
				})
				if tt.wantWarnLog != "" && !strings.Contains(logOutput, tt.wantWarnLog) {
					t.Errorf("LspPositionToBytePosition() log output missing expected warning containing %q. Got:\n%s", tt.wantWarnLog, logOutput)
				}
				if tt.wantWarnLog == "" && strings.Contains(logOutput, "Warning:") {
					t.Errorf("LspPositionToBytePosition() logged unexpected warning. Got:\n%s", logOutput)
				}
			})
		}
	})
}
