// deepcomplete/deepcomplete_test.go
package deepcomplete

import (
	"bytes"
	"context"
	"errors"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"reflect" // For comparing complex structs/slices
	"strings"
	"testing"
	// For spinner test
)

// Helper function to create a FileSet and File for testing
func createFileSetAndFile(t *testing.T, name, content string) (*token.FileSet, *token.File) {
	t.Helper()
	fset := token.NewFileSet()
	file := fset.AddFile(name, fset.Base(), len(content))
	if file == nil {
		t.Fatalf("Failed to add file '%s' to fileset", name)
	}
	return fset, file
}

// TestCalculateCursorPos tests the conversion of line/column to token.Pos offset.
func TestCalculateCursorPos(t *testing.T) {
	// Sample file content for position calculation
	content := `package main

import "fmt"

func main() {
	fmt.Println("Hello") // Line 5
} // Line 6
// Line 7
`
	fset, file := createFileSetAndFile(t, "test_calcpos.go", content)

	tests := []struct {
		name       string
		line       int
		col        int
		wantPos    token.Pos // Expected token.Pos (offset based)
		wantErr    bool
		wantOffset int // Expected byte offset for verification
	}{
		{"Start of file", 1, 1, file.Pos(0), false, 0},
		{"Start of line 3", 3, 1, file.Pos(15), false, 15},            // Offset of 'import "fmt"'
		{"Middle of Println", 5, 10, file.Pos(45), false, 45},         // Offset within Println
		{"End of Println line", 5, 25, file.Pos(60), false, 60},       // Offset after ')' on line 5
		{"Cursor after Println line", 5, 26, file.Pos(61), false, 61}, // Offset at newline char after line 5
		{"Start of line 6", 6, 1, file.Pos(62), false, 62},            // Offset of '}' on line 6
		{"End of line 6", 6, 2, file.Pos(63), false, 63},              // Offset after '}' on line 6
		{"End of file", 7, 10, file.Pos(73), false, 73},               // Offset at end of content
		{"Cursor after end of file", 7, 11, file.Pos(73), false, 73},  // Clamped to end of file size
		{"Invalid line (too high)", 8, 1, token.NoPos, true, -1},
		{"Invalid line (zero)", 0, 1, token.NoPos, true, -1},
		{"Invalid col (zero)", 5, 0, token.NoPos, true, -1},
		// Updated expected behavior for column too high - clamped to end of line offset + 1 (newline offset)
		{"Invalid col (too high for line)", 5, 100, file.Pos(61), false, 61}, // Clamped to end of line + 1 (newline offset)
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
					t.Errorf("calculateCursorPos() returned invalid pos %v, want valid pos %v", gotPos, tt.wantPos)
				} else {
					// Compare offsets as token.Pos values might differ slightly across runs? Unlikely but safer.
					gotOffset := file.Offset(gotPos)
					if gotOffset != tt.wantOffset {
						t.Errorf("calculateCursorPos() calculated offset = %d, want %d (Pos: got %v, want %v)", gotOffset, tt.wantOffset, gotPos, tt.wantPos)
					}
				}
			}
		})
	}
}

// --- Basic Test for AnalyzeCodeContext (without full package loading) ---

// TestAnalyzeCodeContext_DirectParse tests basic context extraction by directly parsing a string.
// This avoids the complexity of setting up go/packages for simple cases but won't have type info.
func TestAnalyzeCodeContext_DirectParse(t *testing.T) {
	content := `package main

import "fmt" // Some import

// GlobalConst is a global constant.
const GlobalConst = 123

var GlobalVar string // Another global

type MyStruct struct { FieldA int }

// MyFunction is the enclosing function.
// It has two arguments.
func MyFunction(arg1 int, arg2 string) (res int, err error) {
	localVar := "hello" // A local variable
	_ = localVar // Use localVar
	var s MyStruct
	s.FieldA = arg1 // Cursor for selector test -> Line 16 Col 4
	fmt.Println(arg1, ) // Cursor for call test -> Line 17 Col 17
	// Cursor is here -> Line 18 Col 2
} // End of function
`
	// Create a temporary file for analysis (AnalyzeCodeContext expects a file path)
	// Use t.TempDir() for automatic cleanup
	tmpDir := t.TempDir()
	tmpFilename := filepath.Join(tmpDir, "test_analyze.go")
	err := os.WriteFile(tmpFilename, []byte(content), 0644)
	if err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	// Create analyzer instance (using the internal struct directly for testing)
	analyzer := NewGoPackagesAnalyzer() // Use exported constructor

	// --- Test Case: Inside MyFunction (general) ---
	t.Run("Inside MyFunction", func(t *testing.T) {
		line, col := 18, 2 // Cursor position within the function body ("// Cursor is here")
		ctx := context.Background()

		// Call Analyze method - it will use the direct parse fallback as no go.mod exists
		// We expect non-fatal errors related to package loading / type info missing
		info, analysisErr := analyzer.Analyze(ctx, tmpFilename, line, col)

		// Check for fatal errors first (should not happen if file exists and pos is valid)
		if analysisErr != nil && !isExpectedAnalysisError(analysisErr) {
			t.Fatalf("Analyze returned unexpected error: %v", analysisErr)
		}
		if info == nil {
			t.Fatal("Analyze returned nil info")
		}

		// --- Assertions on extracted context ---

		// Check enclosing function (won't have type info, check AST node)
		if info.EnclosingFuncNode == nil {
			t.Errorf("Expected EnclosingFuncNode to be non-nil")
		} else if info.EnclosingFuncNode.Name == nil || info.EnclosingFuncNode.Name.Name != "MyFunction" {
			t.Errorf("Expected enclosing function 'MyFunction', got %v", info.EnclosingFuncNode.Name)
		}

		// Check comments (should find the function doc comment)
		foundDocComment := false
		expectedDoc := "MyFunction is the enclosing function."
		for _, c := range info.CommentsNearCursor {
			cleanComment := strings.TrimSpace(strings.TrimPrefix(c, "//"))
			if strings.Contains(cleanComment, expectedDoc) {
				foundDocComment = true
				break
			}
		}
		if !foundDocComment {
			t.Errorf("Expected function Doc comment containing '%s', got: %v", expectedDoc, info.CommentsNearCursor)
		}

		// Check preamble contains expected elements (basic check)
		if !strings.Contains(info.PromptPreamble, "// Enclosing Function (AST): func MyFunction(...)") {
			t.Errorf("Preamble missing expected enclosing function. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, expectedDoc) {
			t.Errorf("Preamble missing expected doc comment. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, "in Scope") {
			t.Errorf("Preamble missing expected scope section. Got:\n%s", info.PromptPreamble)
		}
		// Check for globals (might appear depending on scope analysis without type info)
		if !strings.Contains(info.PromptPreamble, "//   GlobalConst") {
			t.Logf("Preamble missing GlobalConst (may be expected without type info). Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, "//   GlobalVar") {
			t.Logf("Preamble missing GlobalVar (may be expected without type info). Got:\n%s", info.PromptPreamble)
		}
		// Check Imports
		if !strings.Contains(info.PromptPreamble, "//   import \"fmt\"") {
			t.Errorf("Preamble missing import fmt. Got:\n%s", info.PromptPreamble)
		}
	})

	// --- Test Case: Selector Expression ---
	t.Run("Selector Expression", func(t *testing.T) {
		line, col := 16, 4 // Cursor position after "s."
		ctx := context.Background()
		info, analysisErr := analyzer.Analyze(ctx, tmpFilename, line, col)

		if analysisErr != nil && !isExpectedAnalysisError(analysisErr) {
			t.Fatalf("Analyze returned unexpected error: %v", analysisErr)
		}
		if info == nil {
			t.Fatal("Analyze returned nil info")
		}

		if info.SelectorExpr == nil {
			t.Fatalf("Expected SelectorExpr context, got nil")
		}
		if info.SelectorExpr.Sel == nil || info.SelectorExpr.Sel.Name != "FieldA" {
			t.Errorf("Expected selector 'FieldA', got %v", info.SelectorExpr.Sel)
		}
		if !strings.Contains(info.PromptPreamble, "// Selector context:") {
			t.Errorf("Preamble missing selector context. Got:\n%s", info.PromptPreamble)
		}
		// Cannot reliably test members without type info
	})

	// --- Test Case: Call Expression ---
	t.Run("Call Expression", func(t *testing.T) {
		line, col := 17, 17 // Cursor position after comma in fmt.Println(arg1, |)
		ctx := context.Background()
		info, analysisErr := analyzer.Analyze(ctx, tmpFilename, line, col)

		if analysisErr != nil && !isExpectedAnalysisError(analysisErr) {
			t.Fatalf("Analyze returned unexpected error: %v", analysisErr)
		}
		if info == nil {
			t.Fatal("Analyze returned nil info")
		}

		if info.CallExpr == nil {
			t.Fatalf("Expected CallExpr context, got nil")
		}
		if info.CallArgIndex != 1 {
			t.Errorf("Expected CallArgIndex 1, got %d", info.CallArgIndex)
		}
		// Cannot reliably test CallExprFuncType without type info
		if !strings.Contains(info.PromptPreamble, "// Inside function call:") {
			t.Errorf("Preamble missing call expression context. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, "(Arg 2)") { // Check 1-based index in preamble
			t.Errorf("Preamble missing correct arg index. Got:\n%s", info.PromptPreamble)
		}
	})

}

// isExpectedAnalysisError checks if the error is one of the known non-fatal errors
// that can occur during analysis, especially when type info is missing.
func isExpectedAnalysisError(analysisErr error) bool {
	if analysisErr == nil {
		return true
	}
	errStr := analysisErr.Error()
	expectedSubstrings := []string{
		"package loading failed",
		"package loading reported errors",
		"package type information missing",
		"package scope information missing",
		"Type information unavailable",
		"Type/Scope information unavailable",
		"Skipping", // Ignore type resolution skips
		"type info maps (Uses, Defs, Types) are nil",
		"missing type info",
		"types map missing",
		"object",         // From "object 'X' found but type is nil"
		"internal panic", // From panic recovery
	}
	for _, sub := range expectedSubstrings {
		if strings.Contains(errStr, sub) {
			return true
		}
	}
	// Check joined errors
	if joinedErr, ok := analysisErr.(interface{ Unwrap() []error }); ok {
		allExpected := true
		for _, subErr := range joinedErr.Unwrap() {
			if !isExpectedAnalysisError(subErr) {
				allExpected = false
				break
			}
		}
		return allExpected
	}

	return false
}

func TestFindCommentsWithMap(t *testing.T) {
	// TODO: Implement more focused tests for findCommentsWithMap
	t.Skip("findCommentsWithMap tests not yet implemented")
}

func TestFindIdentifierAtCursor(t *testing.T) {
	// TODO: Implement tests for findIdentifierAtCursor
	t.Skip("findIdentifierAtCursor tests not yet implemented")
}

// TestLoadConfig tests configuration loading from standard locations.
func TestLoadConfig(t *testing.T) {
	// Store original env vars and restore them later
	origConfigDir := os.Getenv("XDG_CONFIG_HOME")
	origHome := os.Getenv("HOME")
	if origHome == "" {
		origHome = os.Getenv("USERPROFILE")
	} // Windows fallback
	t.Cleanup(func() {
		os.Setenv("XDG_CONFIG_HOME", origConfigDir)
		os.Setenv("HOME", origHome)
		os.Setenv("USERPROFILE", origHome)
	})

	// Create temp dir for fake home/config
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)        // Unix-like HOME
	t.Setenv("USERPROFILE", tempHome) // Windows HOME

	// Define config paths within temp dir
	fakeConfigDir := filepath.Join(tempHome, ".config", configDirName)
	fakeDataDir := filepath.Join(tempHome, ".local", "share", configDirName)
	fakeConfigFile := filepath.Join(fakeConfigDir, defaultConfigFileName)
	fakeDataFile := filepath.Join(fakeDataDir, defaultConfigFileName)

	// --- Test Cases ---
	tests := []struct {
		name          string
		setup         func() error // Function to set up files/env for the test
		wantConfig    Config
		checkWrite    bool   // Whether to check if default config was written
		wantWritePath string // Path where write is expected
		wantErrLog    string // Substring of expected warning log, empty if no warning expected
	}{
		{
			name: "No config files - writes default",
			setup: func() error {
				os.RemoveAll(fakeConfigDir) // Ensure dirs don't exist initially
				os.RemoveAll(fakeDataDir)
				t.Setenv("XDG_CONFIG_HOME", filepath.Join(tempHome, ".config")) // Point to temp .config
				return nil
			},
			wantConfig:    DefaultConfig, // Should return defaults
			checkWrite:    true,
			wantWritePath: fakeConfigFile,
			wantErrLog:    "", // No error expected, just info log about writing default
		},
		{
			name: "Config in XDG_CONFIG_HOME",
			setup: func() error {
				os.RemoveAll(fakeDataDir) // Ensure data dir file doesn't exist
				os.MkdirAll(fakeConfigDir, 0755)
				jsonData := `{"model": "test-model-config", "temperature": 0.99, "use_fim": true}`
				t.Setenv("XDG_CONFIG_HOME", filepath.Join(tempHome, ".config"))
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig: Config{
				OllamaURL: DefaultConfig.OllamaURL, Model: "test-model-config",
				PromptTemplate: DefaultConfig.PromptTemplate, FimTemplate: DefaultConfig.FimTemplate,
				MaxTokens: DefaultConfig.MaxTokens, Stop: DefaultConfig.Stop,
				Temperature: 0.99, UseAst: DefaultConfig.UseAst, UseFim: true,
			},
			checkWrite: false, // Should not write default if file exists
			wantErrLog: "",
		},
		{
			name: "Config in .local/share (XDG_CONFIG_HOME unset)",
			setup: func() error {
				os.RemoveAll(fakeConfigDir) // Ensure config dir file doesn't exist
				os.MkdirAll(fakeDataDir, 0755)
				jsonData := `{"ollama_url": "http://otherhost:1111", "stop": ["\n", "stop"], "use_ast": false}`
				t.Setenv("XDG_CONFIG_HOME", "") // Unset XDG_CONFIG_HOME
				return os.WriteFile(fakeDataFile, []byte(jsonData), 0644)
			},
			wantConfig: Config{
				OllamaURL: "http://otherhost:1111", Model: DefaultConfig.Model,
				PromptTemplate: DefaultConfig.PromptTemplate, FimTemplate: DefaultConfig.FimTemplate,
				MaxTokens: DefaultConfig.MaxTokens, Stop: []string{"\n", "stop"},
				Temperature: DefaultConfig.Temperature, UseAst: false, UseFim: DefaultConfig.UseFim,
			},
			checkWrite: false,
			wantErrLog: "",
		},
		{
			name: "Invalid JSON - writes default",
			setup: func() error {
				os.RemoveAll(fakeDataDir)
				os.MkdirAll(fakeConfigDir, 0755)
				jsonData := `{"model": "bad json",` // Invalid JSON
				t.Setenv("XDG_CONFIG_HOME", filepath.Join(tempHome, ".config"))
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig:    DefaultConfig, // Should return defaults on parse error
			checkWrite:    true,          // Should still write default if parse fails
			wantWritePath: fakeConfigFile,
			wantErrLog:    "parsing config file JSON failed", // Expect warning log
		},
		// Add more cases: precedence (XDG over .local), partial configs, different fields
	}

	// --- Run Tests ---
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.setup(); err != nil {
				t.Fatalf("Setup failed: %v", err)
			}
			defer os.RemoveAll(fakeConfigDir) // Ensure cleanup
			defer os.RemoveAll(fakeDataDir)

			// Capture log output
			var logBuf bytes.Buffer
			log.SetOutput(&logBuf)
			defer log.SetOutput(os.Stderr) // Restore default logger

			gotConfig, err := LoadConfig() // Call the function under test

			// LoadConfig itself doesn't return parse/write errors directly, only logs them
			if err != nil {
				// Check if the error contains expected non-fatal config load/write issues
				errStr := err.Error()
				isExpectedLoadErr := strings.Contains(errStr, "loading") || strings.Contains(errStr, "writing") || strings.Contains(errStr, "parsing") || strings.Contains(errStr, "cannot find") || strings.Contains(errStr, "Could not determine")
				if !isExpectedLoadErr {
					t.Errorf("LoadConfig() returned unexpected error = %v", err)
				} else {
					t.Logf("LoadConfig() returned expected non-fatal error: %v", err)
				}
			}

			// Check log output for expected warnings
			logOutput := logBuf.String()
			if tt.wantErrLog != "" && !strings.Contains(logOutput, tt.wantErrLog) {
				t.Errorf("LoadConfig() log output missing expected warning containing %q. Got:\n%s", tt.wantErrLog, logOutput)
			}
			if tt.wantErrLog == "" && strings.Contains(logOutput, "Warning:") && !strings.Contains(logOutput, "Could not determine user") && !strings.Contains(logOutput, "writing default config") {
				// Allow warnings about missing user dirs and writing defaults, but not others if no error expected
				t.Errorf("LoadConfig() logged unexpected warning. Got:\n%s", logOutput)
			}

			// Use reflect.DeepEqual for struct comparison, ignoring templates
			tempWant := tt.wantConfig
			tempGot := gotConfig
			tempWant.PromptTemplate = ""
			tempGot.PromptTemplate = ""
			tempWant.FimTemplate = ""
			tempGot.FimTemplate = ""

			if !reflect.DeepEqual(tempGot, tempWant) {
				t.Errorf("LoadConfig() got = %+v, want %+v", gotConfig, tt.wantConfig)
			}

			// Check if default file was written if expected
			if tt.checkWrite {
				if _, statErr := os.Stat(tt.wantWritePath); errors.Is(statErr, os.ErrNotExist) {
					t.Errorf("LoadConfig() did not write default config file to %s when expected", tt.wantWritePath)
				} else if statErr != nil {
					t.Errorf("Error checking for default config file %s: %v", tt.wantWritePath, statErr)
				} else {
					// Optionally read the written file and verify its content matches defaults
					// content, readErr := os.ReadFile(tt.wantWritePath) ... json.Unmarshal ... DeepEqual ...
				}
			} else {
				// Ensure file wasn't written if not expected (e.g., when valid config existed)
				// Check both potential paths as we don't know which one might have been tried if setup failed partially
				if _, statErr := os.Stat(fakeConfigFile); statErr == nil && tt.name != "Config in XDG_CONFIG_HOME" && tt.name != "Invalid JSON - writes default" {
					t.Logf("Warning: Default config file %s exists when not expected (might be from previous test)", fakeConfigFile)
				}
				if _, statErr := os.Stat(fakeDataFile); statErr == nil && tt.name != "Config in .local/share (XDG_CONFIG_HOME unset)" {
					t.Logf("Warning: Default config file %s exists when not expected (might be from previous test)", fakeDataFile)
				}
			}
		})
	}
}

// TestExtractSnippetContext tests prefix/suffix extraction.
func TestExtractSnippetContext(t *testing.T) {
	content := `line one
line two
line three`
	tmpDir := t.TempDir()
	tmpFilename := filepath.Join(tmpDir, "test_snippet.go")
	err := os.WriteFile(tmpFilename, []byte(content), 0644)
	if err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	tests := []struct {
		name       string
		line       int
		col        int
		wantPrefix string
		wantSuffix string
		wantLine   string
		wantErr    bool
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
					t.Errorf("extractSnippetContext() Prefix = %q, want %q", gotCtx.Prefix, tt.wantPrefix)
				}
				if gotCtx.Suffix != tt.wantSuffix {
					t.Errorf("extractSnippetContext() Suffix = %q, want %q", gotCtx.Suffix, tt.wantSuffix)
				}
				if gotCtx.FullLine != tt.wantLine {
					t.Errorf("extractSnippetContext() FullLine = %q, want %q", gotCtx.FullLine, tt.wantLine)
				}
			}
		})
	}
}

// TestListTypeMembers tests listing fields/methods.
func TestListTypeMembers(t *testing.T) {
	// TODO: Implement tests for listTypeMembers
	// Requires setting up mock types.Type or using type checker on test code.
	// This is complex to set up correctly without loading real packages.
	t.Skip("listTypeMembers tests not yet implemented")
}

func TestFindCommentsWithMap(t *testing.T) {
	// TODO: Implement more focused tests for findCommentsWithMap
	t.Skip("findCommentsWithMap tests not yet implemented")
}

func TestFindIdentifierAtCursor(t *testing.T) {
	// TODO: Implement tests for findIdentifierAtCursor
	t.Skip("findIdentifierAtCursor tests not yet implemented")
}
