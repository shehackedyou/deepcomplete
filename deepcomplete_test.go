// deepcomplete/deepcomplete_test.go
package deepcomplete

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"reflect" // For comparing complex structs/slices
	"strings"
	"testing"
	"time"

	"go.etcd.io/bbolt"
	// For spinner test
)

// Helper function to create a FileSet and File for testing
func createFileSetAndFile(t *testing.T, name, content string) (*token.FileSet, *token.File) {
	t.Helper()
	fset := token.NewFileSet()
	// Use non-zero base for safety, although 0 often works.
	// Using 1 as base, ensuring file size matches content length.
	file := fset.AddFile(name, 1, len(content))
	if file == nil {
		t.Fatalf("Failed to add file '%s' to fileset", name)
	}
	// Note: Setting lines manually is fragile if content changes.
	// Rely on fset.Position calculations instead.
	// file.SetLinesForContent([]byte(content)) // This might be needed if using line info directly

	return fset, file
}

// TestCalculateCursorPos tests the conversion of line/column to token.Pos offset.
// (Cycle 1 Change: Added multi-byte test case)
func TestCalculateCursorPos(t *testing.T) {
	// Sample file content for position calculation
	content := `package main

import "fmt" // Line 3

func main() { // Line 5
	fmt.Println("Hello") // Line 6
	fmt.Println("你好世界") // Line 7 (Multi-byte)
} // Line 8
// Line 9` // Ensure content ends without newline for some tests

	// Use a simple fileset for this test, file content doesn't need full parsing here
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
		wantOffset int // Expected 0-based byte offset for verification
	}{
		{"Start of file", 1, 1, false, 0},
		{"Start of line 3", 3, 1, false, 15},                         // Offset of 'import "fmt"'
		{"Middle of Println", 6, 10, false, 46},                      // Offset within Println "Hello"
		{"End of Println line", 6, 26, false, 62},                    // Offset after ')' on line 6
		{"Cursor after Println line (at newline)", 6, 27, false, 62}, // Clamped to line end offset (before newline)
		{"Start of line 7", 7, 1, false, 63},                         // Offset of 'f' in fmt
		{"Middle of multi-byte string (before 好)", 7, 16, false, 78}, // Offset inside "你好世界" quotes, before 好
		{"Middle of multi-byte string (after 好)", 7, 17, false, 81},  // Offset inside "你好世界" quotes, after 好 (3 bytes)
		{"Middle of multi-byte string (after 界)", 7, 20, false, 90},  // Offset inside "你好世界" quotes, after 界 (3 bytes)
		{"End of multi-byte string line", 7, 23, false, 93},          // Offset after ')' on line 7
		{"Start of line 8", 8, 1, false, 94},                         // Offset of '}' on line 8
		{"End of line 8", 8, 2, false, 95},                           // Offset after '}' on line 8
		{"Start of line 9", 9, 1, false, 96},                         // Offset of '/' in comment
		{"End of line 9", 9, 10, false, 105},                         // Offset at end of content
		{"Cursor after end of file", 9, 11, false, 105},              // Clamped to end of file size
		{"Invalid line (too high)", 10, 1, true, -1},
		{"Invalid line (zero)", 0, 1, true, -1},
		{"Invalid col (zero)", 6, 0, true, -1},
		{"Invalid col (too high for line)", 6, 100, false, 62}, // Clamped to end of line offset
		{"Col too high last line", 9, 100, false, 105},         // Clamped to end of file offset
		{"Col too high last line + 1", 10, 100, true, -1},      // Invalid line number
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Need to re-add file for each subtest if fileset/file state matters across calls? Unlikely here.
			// Ensure file lines are set correctly if not automatic
			// file.SetLinesForContent([]byte(content)) // Potentially needed per subtest run?

			gotPos, err := calculateCursorPos(file, tt.line, tt.col)

			if (err != nil) != tt.wantErr {
				t.Errorf("calculateCursorPos() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if !gotPos.IsValid() {
					// Calculate expected Pos for comparison message
					wantPos := file.Pos(tt.wantOffset)
					t.Errorf("calculateCursorPos() returned invalid pos %v, want valid pos %v (offset %d)", gotPos, wantPos, tt.wantOffset)
				} else {
					// Compare offsets as primary check
					gotOffset := file.Offset(gotPos)
					if gotOffset != tt.wantOffset {
						t.Errorf("calculateCursorPos() calculated offset = %d, want %d (Pos: got %v, want %v)", gotOffset, tt.wantOffset, gotPos, file.Pos(tt.wantOffset))
					}
				}
			} else {
				// Check if NoPos is returned on error, as expected
				if gotPos.IsValid() {
					t.Errorf("calculateCursorPos() returned valid pos %v on error, want invalid pos", gotPos)
				}
			}
		})
	}
}

// --- Basic Test for AnalyzeCodeContext (without full package loading) ---

// TestAnalyzeCodeContext_DirectParse tests basic context extraction by directly parsing a string.
// This avoids the complexity of setting up go/packages for simple cases but won't have type info.
// NOTE: With the cache change, this test might behave differently if caching were enabled,
// as a cache hit would skip analysis steps. Caching is disabled by default in tests unless explicitly set up.
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
	// Explicitly disable caching for this direct parse test by not providing a DB path
	// (NewGoPackagesAnalyzer handles nil DB gracefully)
	analyzer := NewGoPackagesAnalyzer() // Will create without DB path
	// Ensure analyzer resources are closed
	t.Cleanup(func() {
		t.Log("Closing analyzer in test cleanup...") // Add log for verification
		if err := analyzer.Close(); err != nil {
			t.Errorf("Error closing analyzer: %v", err)
		}
	})

	// --- Test Case: Inside MyFunction (general) ---
	t.Run("Inside MyFunction", func(t *testing.T) {
		line, col := 18, 2 // Cursor position within the function body ("// Cursor is here")
		ctx := context.Background()

		// Call Analyze method - it will use packages.Load (no cache involved here)
		// We expect non-fatal errors related to package loading / type info missing
		// because there's no go.mod in the temp dir.
		info, analysisErr := analyzer.Analyze(ctx, tmpFilename, line, col)

		// Check for fatal errors first (should not happen if file exists and pos is valid)
		isNonFatalLoadErr := analysisErr != nil && errors.Is(analysisErr, ErrAnalysisFailed)

		if analysisErr != nil && !isNonFatalLoadErr {
			// Unexpected fatal error
			t.Fatalf("Analyze returned unexpected fatal error: %v", analysisErr)
		}
		if info == nil {
			t.Fatal("Analyze returned nil info")
		}
		// Log expected non-fatal errors if they occurred
		if isNonFatalLoadErr {
			t.Logf("Analyze returned expected non-fatal errors: %v", analysisErr)
			// Verify specific expected errors are present in info.AnalysisErrors if needed
			foundLoadError := false
			for _, e := range info.AnalysisErrors {
				// Check for errors indicating type info issues or load failures
				// which are expected when direct parsing without go/packages context
				errStr := e.Error()
				if strings.Contains(errStr, "type info missing") ||
					strings.Contains(errStr, "package loading failed") ||
					strings.Contains(errStr, "type info unavailable") ||
					strings.Contains(errStr, "TypesInfo is nil") ||
					strings.Contains(errStr, "Types is nil") || // Check for nil types map
					strings.Contains(errStr, "missing object info") || // Check for missing object lookups
					strings.Contains(errStr, "package scope missing") || // Check for scope issues
					strings.Contains(errStr, "TypesInfo") { // General check for TypeInfo problems
					foundLoadError = true
					break
				}
			}
			if !foundLoadError {
				t.Errorf("Expected package load/type info errors in info.AnalysisErrors, got: %v", info.AnalysisErrors)
			}
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
		// Updated expected string based on previous changes (AST only, unknown types)
		if !strings.Contains(info.PromptPreamble, "// Enclosing Function (AST only): MyFunction(...)") {
			t.Errorf("Preamble missing expected enclosing function. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, expectedDoc) {
			t.Errorf("Preamble missing expected doc comment. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, "Variables/Constants/Types in Scope") { // Check generic scope header
			t.Errorf("Preamble missing expected scope section. Got:\n%s", info.PromptPreamble)
		}
		// Check for globals (might appear depending on scope analysis without type info)
		// Note: Without type info, scope resolution is limited. Check if they appear at all.
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
		// Check AST details
		if selX, ok := info.SelectorExpr.X.(*ast.Ident); !ok || selX.Name != "s" {
			t.Errorf("Expected selector base 's', got %T", info.SelectorExpr.X)
		}
		if info.SelectorExpr.Sel == nil || info.SelectorExpr.Sel.Name != "FieldA" {
			t.Errorf("Expected selector identifier 'FieldA', got %v", info.SelectorExpr.Sel)
		}

		// Check preamble (type info will be missing/unknown, fallback added in Step 10)
		if !strings.Contains(info.PromptPreamble, "// Selector context: expr type = (unknown - type analysis failed") {
			t.Errorf("Preamble missing selector context or expected fallback message. Got:\n%s", info.PromptPreamble)
		}
		// Member listing should show failure message (fallback added in Step 10)
		if !strings.Contains(info.PromptPreamble, "//   (Cannot list members: type analysis failed") {
			t.Errorf("Preamble missing expected member listing fallback. Got:\n%s", info.PromptPreamble)
		}
	})

	// --- Test Case: Call Expression ---
	t.Run("Call Expression", func(t *testing.T) {
		line, col := 17, 17 // Cursor position after comma in fmt.Println(arg1, |)
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
		if info.CallArgIndex != 1 { // Arg index calculation should still work
			t.Errorf("Expected CallArgIndex 1, got %d", info.CallArgIndex)
		}
		// Check preamble (type info will be missing/unknown, fallback added in Step 10)
		if !strings.Contains(info.PromptPreamble, "// Inside function call: Println (Arg 2)") { // Check func name from AST, 1-based index
			t.Errorf("Preamble missing call expression context. Got:\n%s", info.PromptPreamble)
		}
		// Check for fallback message for signature (Step 10)
		if !strings.Contains(info.PromptPreamble, "// Function Signature: (unknown - type analysis failed") {
			t.Errorf("Preamble missing expected unknown signature fallback. Got:\n%s", info.PromptPreamble)
		}
	})

}

// TestLoadConfig tests configuration loading from standard locations.
func TestLoadConfig(t *testing.T) {
	// Store original env vars and restore them later
	origConfigDir := os.Getenv("XDG_CONFIG_HOME")
	origHome := os.Getenv("HOME")
	origUserProfile := os.Getenv("USERPROFILE") // Windows fallback
	t.Cleanup(func() {
		os.Setenv("XDG_CONFIG_HOME", origConfigDir)
		os.Setenv("HOME", origHome)
		os.Setenv("USERPROFILE", origUserProfile)
	})

	// Create temp dir for fake home/config
	tempHome := t.TempDir() // t.Cleanup handles removal
	// Explicitly set env vars for consistent paths across OSes
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(tempHome, ".config"))
	t.Setenv("HOME", tempHome)
	t.Setenv("USERPROFILE", tempHome)

	// Define config paths within temp dir (using XDG path as primary)
	fakeConfigDir := filepath.Join(tempHome, ".config", configDirName)
	// Note: Secondary path check might be needed if XDG fails AND home works.
	// fakeDataDir := filepath.Join(tempHome, ".local", "share", configDirName)
	fakeConfigFile := filepath.Join(fakeConfigDir, defaultConfigFileName)
	// fakeDataFile := filepath.Join(fakeDataDir, defaultConfigFileName)

	// --- Test Cases ---
	tests := []struct {
		name          string
		setup         func(t *testing.T) error // Function to set up files/env for the test
		wantConfig    Config
		checkWrite    bool   // Whether to check if default config was written
		wantWritePath string // Path where write is expected
		wantErrLog    string // Substring of expected warning log, empty if no warning expected
	}{
		{
			name: "No config files - writes default",
			setup: func(t *testing.T) error {
				// Ensure dirs don't exist initially by virtue of t.TempDir() being clean
				// Also remove potential leftovers from previous failed runs if needed
				os.RemoveAll(fakeConfigDir)
				return nil
			},
			wantConfig:    DefaultConfig, // Should return defaults
			checkWrite:    true,
			wantWritePath: fakeConfigFile, // Expect write to primary XDG path
			wantErrLog:    "",             // No error expected, just info log about writing default
		},
		{
			name: "Config in XDG_CONFIG_HOME",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				// Use different values than defaults, including new fields
				jsonData := `{"model": "test-model-config", "temperature": 0.99, "use_fim": true, "max_preamble_len": 1024, "max_snippet_len": 1000}`
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig: Config{
				OllamaURL: DefaultConfig.OllamaURL, Model: "test-model-config",
				PromptTemplate: DefaultConfig.PromptTemplate, FimTemplate: DefaultConfig.FimTemplate,
				MaxTokens: DefaultConfig.MaxTokens, Stop: DefaultConfig.Stop,
				Temperature: 0.99, UseAst: DefaultConfig.UseAst, UseFim: true,
				MaxPreambleLen: 1024, MaxSnippetLen: 1000, // Check new fields are loaded
			},
			checkWrite: false, // Should not write default if file exists
			wantErrLog: "",
		},
		// Note: Testing secondary path (~/.local/share) is harder now that primary uses HOME fallback.
		// We assume getConfigPaths logic handles finding the best available path.
		{
			name: "Invalid JSON - returns defaults, logs warning, writes default",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				jsonData := `{"model": "bad json",` // Invalid JSON
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig:    DefaultConfig, // Should return defaults on parse error
			checkWrite:    true,          // Should still write default if parse fails
			wantWritePath: fakeConfigFile,
			wantErrLog:    "parsing config file JSON", // Expect warning log about parsing
		},
		{
			name: "Partial Config - merges with defaults",
			setup: func(t *testing.T) error {
				if err := os.MkdirAll(fakeConfigDir, 0755); err != nil {
					return err
				}
				jsonData := `{"ollama_url": "http://other:1111", "use_ast": false, "max_snippet_len": 4096}` // Mix old and new
				return os.WriteFile(fakeConfigFile, []byte(jsonData), 0644)
			},
			wantConfig: Config{
				OllamaURL: "http://other:1111", Model: DefaultConfig.Model,
				PromptTemplate: DefaultConfig.PromptTemplate, FimTemplate: DefaultConfig.FimTemplate,
				MaxTokens: DefaultConfig.MaxTokens, Stop: DefaultConfig.Stop,
				Temperature: DefaultConfig.Temperature, UseAst: false, UseFim: DefaultConfig.UseFim,
				MaxPreambleLen: DefaultConfig.MaxPreambleLen, MaxSnippetLen: 4096, // Check merge
			},
			checkWrite: false,
			wantErrLog: "",
		},
	}

	// --- Run Tests ---
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Rerun setup for each test
			if err := tt.setup(t); err != nil {
				t.Fatalf("Setup failed: %v", err)
			}
			// Note: TempDir cleanup is handled automatically by t.Cleanup()

			// Capture log output
			var logBuf bytes.Buffer
			log.SetOutput(&logBuf)
			t.Cleanup(func() { log.SetOutput(os.Stderr) }) // Restore default logger

			gotConfig, err := LoadConfig() // Call the function under test

			// Check for unexpected fatal errors (ErrConfig itself is non-fatal here)
			// Also allow ErrInvalidConfig as LoadConfig now validates before returning
			if err != nil && !errors.Is(err, ErrConfig) && !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("LoadConfig() returned unexpected fatal error = %v", err)
			} else if err != nil {
				t.Logf("LoadConfig() returned expected non-fatal error: %v", err)
			}

			// Check log output for expected warnings
			logOutput := logBuf.String()
			if tt.wantErrLog != "" && !strings.Contains(logOutput, tt.wantErrLog) {
				t.Errorf("LoadConfig() log output missing expected warning containing %q. Got:\n%s", tt.wantErrLog, logOutput)
			}
			// Be more lenient about warnings if an error was expected/returned
			if tt.wantErrLog == "" && err == nil {
				if strings.Contains(logOutput, "Warning:") && !strings.Contains(logOutput, "writing default config") && !strings.Contains(logOutput, "Could not determine") && !strings.Contains(logOutput, "validation failed") {
					// Allow warnings about writing defaults, missing user dirs, and validation fallbacks, but not others if no error expected
					t.Errorf("LoadConfig() logged unexpected warning. Got:\n%s", logOutput)
				}
			}

			// Use reflect.DeepEqual for struct comparison, ignoring templates
			tempWant := tt.wantConfig
			tempGot := gotConfig
			tempWant.PromptTemplate = "" // Ignore internal fields
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
					t.Logf("Default config file written correctly to %s", tt.wantWritePath)
				}
			} else {
				// Ensure file wasn't written if not expected (e.g., when valid config existed)
				if _, statErr := os.Stat(tt.wantWritePath); statErr == nil && !strings.Contains(tt.name, "writes default") {
					// Check if file exists unexpectedly
					t.Errorf("LoadConfig() wrote default config file %s when NOT expected", tt.wantWritePath)
				}
			}
		})
	}
}

// TestExtractSnippetContext tests prefix/suffix extraction.
func TestExtractSnippetContext(t *testing.T) {
	content := `line one
line two
line three` // No newline at end
	tmpDir := t.TempDir() // Auto cleanup
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
		{"End of line 1 (before newline)", 1, 9, "line one", "\nline two\nline three", "line one", false},
		{"Start of line 2", 2, 1, "line one\n", "line two\nline three", "line two", false},
		{"Middle of line 2", 2, 6, "line one\nline ", "two\nline three", "line two", false},
		{"End of line 3 (end of file)", 3, 11, "line one\nline two\nline three", "", "line three", false},
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

// --- Tests needing more work ---

// TestListTypeMembers tests listing fields/methods.
func TestListTypeMembers(t *testing.T) {
	// TODO: Implement tests for listTypeMembers (Refactored Step 5)
	// Requires setting up mock types.Type or using type checker on test code.
	// This is complex to set up correctly without loading real packages.
	t.Skip("listTypeMembers tests not yet implemented (requires mock types or type checker setup)")
}

func TestFindCommentsWithMap(t *testing.T) {
	// TODO: Implement more focused tests for findCommentsWithMap
	// Needs AST setup and comment maps.
	t.Skip("findCommentsWithMap tests not yet implemented")
}

func TestFindIdentifierAtCursor(t *testing.T) {
	// TODO: Implement tests for findIdentifierAtCursor
	// Needs AST setup and type info potentially.
	t.Skip("findIdentifierAtCursor tests not yet implemented")
}

// TestGoPackagesAnalyzer_Cache tests the bbolt caching mechanism.
// (Cycle 2 Change: Implemented InvalidateCache test)
func TestGoPackagesAnalyzer_Cache(t *testing.T) {
	// Test InvalidateCache specifically
	t.Run("InvalidateCache", func(t *testing.T) {
		// Setup: Create temp dir, temp file, temp cache DB
		tmpDir := t.TempDir()
		testFilename := filepath.Join(tmpDir, "cache_test.go")
		testContent := `package main
func main() { println("hello") } // Line 2, Col 20
`
		if err := os.WriteFile(testFilename, []byte(testContent), 0644); err != nil {
			t.Fatalf("Failed to write test file: %v", err)
		}

		// --- Create Analyzer with a temporary DB path ---
		// Override user cache dir logic for isolated test DB
		dbDir := filepath.Join(tmpDir, "test_bboltdb", fmt.Sprintf("v%d", cacheSchemaVersion))
		if err := os.MkdirAll(dbDir, 0750); err != nil {
			t.Fatalf("Failed to create temp db dir: %v", err)
		}
		dbPath := filepath.Join(dbDir, "test_cache.db")

		// Manually create analyzer pointing to test DB path
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
		analyzer := &GoPackagesAnalyzer{db: db} // Inject the test DB
		t.Cleanup(func() {
			t.Log("Closing test analyzer...")
			if err := analyzer.Close(); err != nil { // Close should close the injected DB
				t.Errorf("Error closing test analyzer: %v", err)
			}
			// db file is cleaned up by t.TempDir()
		})
		// ---

		ctx := context.Background()
		line, col := 2, 20

		// --- Capture log output ---
		var logBuf bytes.Buffer
		// Use a test-specific logger temporarily
		testLogger := log.New(&logBuf, "TESTCACHE: ", log.Lshortfile)
		// Swap std logger, restore after test
		oldLogger := log.Default() // Get current default logger
		log.SetOutput(testLogger.Writer())
		log.SetFlags(testLogger.Flags())
		log.SetPrefix(testLogger.Prefix())
		t.Cleanup(func() {
			// Restore previous logger settings
			log.SetOutput(oldLogger.Writer())
			log.SetFlags(oldLogger.Flags())
			log.SetPrefix(oldLogger.Prefix())
		})
		// ---

		// 1. First Analysis (should be a cache miss and populate cache)
		t.Log("Performing first analysis (expect cache miss)...")
		logBuf.Reset() // Clear buffer before first analysis
		_, err = analyzer.Analyze(ctx, testFilename, line, col)
		if err != nil && !errors.Is(err, ErrAnalysisFailed) { // Allow non-fatal analysis errors
			t.Fatalf("First Analyze failed unexpectedly: %v", err)
		}
		logOutput1 := logBuf.String()
		if !strings.Contains(logOutput1, "Cache miss or invalid") && !strings.Contains(logOutput1, "Cache disabled") {
			t.Errorf("Expected 'Cache miss' or 'Cache disabled' log during first analysis, got:\n%s", logOutput1)
		}
		if !strings.Contains(logOutput1, "Saved analysis results to bbolt cache") {
			t.Errorf("Expected 'Saved analysis results' log during first analysis, got:\n%s", logOutput1)
		}

		// 2. Second Analysis (should be a cache hit)
		t.Log("Performing second analysis (expect cache hit)...")
		logBuf.Reset() // Clear buffer
		_, err = analyzer.Analyze(ctx, testFilename, line, col)
		if err != nil && !errors.Is(err, ErrAnalysisFailed) {
			t.Fatalf("Second Analyze failed unexpectedly: %v", err)
		}
		logOutput2 := logBuf.String()
		if !strings.Contains(logOutput2, "Potential cache hit") || !strings.Contains(logOutput2, "Cache VALID") {
			t.Errorf("Expected 'Potential cache hit' and 'Cache VALID' logs during second analysis, got:\n%s", logOutput2)
		}
		if strings.Contains(logOutput2, "Cache miss or invalid") {
			t.Errorf("Unexpected 'Cache miss' log during second analysis (should have hit), got:\n%s", logOutput2)
		}

		// 3. Invalidate Cache
		t.Log("Invalidating cache...")
		logBuf.Reset()
		err = analyzer.InvalidateCache(tmpDir) // Invalidate for the directory
		if err != nil {
			t.Fatalf("InvalidateCache failed: %v", err)
		}
		logOutputInvalidate := logBuf.String()
		if !strings.Contains(logOutputInvalidate, "Invalidating cache for key") || !strings.Contains(logOutputInvalidate, "Deleting cache entry") {
			t.Errorf("Expected 'Invalidating cache' and 'Deleting cache entry' logs, got:\n%s", logOutputInvalidate)
		}

		// 4. Third Analysis (should be a cache miss again)
		t.Log("Performing third analysis (expect cache miss after invalidation)...")
		logBuf.Reset()
		_, err = analyzer.Analyze(ctx, testFilename, line, col)
		if err != nil && !errors.Is(err, ErrAnalysisFailed) {
			t.Fatalf("Third Analyze failed unexpectedly: %v", err)
		}
		logOutput3 := logBuf.String()
		if !strings.Contains(logOutput3, "Cache miss or invalid") {
			t.Errorf("Expected 'Cache miss' log during third analysis after invalidation, got:\n%s", logOutput3)
		}
		if strings.Contains(logOutput3, "Cache VALID") {
			t.Errorf("Unexpected 'Cache VALID' log during third analysis (should have missed), got:\n%s", logOutput3)
		}
		if !strings.Contains(logOutput3, "Saved analysis results to bbolt cache") {
			t.Errorf("Expected 'Saved analysis results' log during third analysis, got:\n%s", logOutput3)
		}
	})

	// TODO: Add tests for cache hits, different file modifications causing misses,
	// go.mod changes causing misses, schema version changes causing misses, etc.
	t.Skip("Remaining GoPackagesAnalyzer caching tests not yet implemented")
}

// ============================================================================
// LSP Position Conversion Tests (Added in Step 1)
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
			wantErrType    error // Specific error type to check
		}{
			// Basic ASCII
			{"ASCII start", "hello", 0, 0, false, nil},
			{"ASCII middle", "hello", 2, 2, false, nil},
			{"ASCII end", "hello", 5, 5, false, nil},
			{"ASCII past end", "hello", 6, 5, true, ErrPositionOutOfRange},    // Error expected, offset clamped by wantByteOffset
			{"ASCII negative", "hello", -1, 0, true, ErrInvalidPositionInput}, // Error expected

			// Multi-byte UTF-8 (2 bytes)
			{"2byte UTF-8 start", "héllo", 0, 0, false, nil},
			{"2byte UTF-8 before", "héllo", 1, 1, false, nil}, // Before 'é'
			{"2byte UTF-8 after", "héllo", 2, 3, false, nil},  // After 'é' (1 UTF-16 unit, 2 bytes)
			{"2byte UTF-8 middle", "héllo", 4, 5, false, nil}, // Before 'o'
			{"2byte UTF-8 end", "héllo", 5, 6, false, nil},
			{"2byte UTF-8 past end", "héllo", 6, 6, true, ErrPositionOutOfRange},

			// Multi-byte UTF-8 (3 bytes)
			{"3byte UTF-8 start", "€ euro", 0, 0, false, nil},
			{"3byte UTF-8 after", "€ euro", 1, 3, false, nil}, //   After '€' (1 UTF-16 unit, 3 bytes)
			{"3byte UTF-8 middle", "€ euro", 3, 5, false, nil}, // Before 'e'
			{"3byte UTF-8 end", "€ euro", 6, 8, false, nil},
			{"3byte UTF-8 past end", "€ euro", 7, 8, true, ErrPositionOutOfRange},

			// Multi-byte UTF-8 (4 bytes - Surrogate Pair in UTF-16)
			{"4byte UTF-8 start", "😂笑", 0, 0, false, nil},
			{"4byte UTF-8 middle (within surrogate)", "😂笑", 1, 0, false, nil}, // Offset 1 falls *within* the 2 UTF-16 units of 😂 -> byte offset remains 0
			{"4byte UTF-8 after surrogate", "😂笑", 2, 4, false, nil},          // Af ter 😂 (2 UTF-16 units, 4 bytes)
			{"4byte UTF-8 after second char", "😂笑", 3, 7, false, nil},         // After 笑 (1 UTF-16 unit, 3 bytes) -> total 4+3=7 bytes
			{"4byte UTF-8 end", "😂笑", 3, 7, false, nil},                       // End is same as after second char for UTF-16 offset
			{"4byte UTF-8 past end", "😂笑", 4, 7, true, ErrPositionOutOfRange},

			// Mixed
			{"Mixed start", "a é 😂 €", 0, 0, false, nil},
			{"Mixed after a", "a é 😂 €", 1, 1, false, nil},     // a (   1b, 1u16)
			{"Mixed after space", "a é 😂 €", 2, 2, false, nil}, // spa   ce (1b, 1u16)
			{"Mixed after é", "a é 😂 €", 3, 4, false, nil},     // é (2   b, 1u16) -> 1+1+2=4 bytes
			{"Mixed after space 2", "a é 😂 €", 4, 5, false, nil}, // sp ace (1b, 1u16) -> 4+1=5 bytes
			{"Mixed within 😂", "a é 😂 €", 5, 5, false, nil},    // Wi   thin 😂 surrogate (4b, 2u16) -> byte offset remains 5
			{"Mixed after 😂", "a é 😂 €", 6, 9, false, nil},     // A   fter 😂 -> 5+4=9 bytes
			{"Mixed after space 3", "a é 😂 €", 7, 10, false, nil},// spa ce (1b, 1u16) -> 9+1=10 bytes
			{"Mixed after €", "a é 😂 €", 8, 13, false, nil},    // A   fter € (3b, 1u16) -> 10+3=13 bytes
			{"Mixed end", "a é 😂 €", 8, 13, false, nil},           // End is same offset
			{"Mixed past end", "a é 😂 €", 9, 13, true, ErrPositionOutOfRange},

			// Empty line
			{"Empty line start", "", 0, 0, false, nil},
			{"Empty line past end", "", 1, 0, true, ErrPositionOutOfRange},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				gotByteOffset, err := Utf16OffsetToBytes([]byte(tt.lineContent), tt.utf16Offset)

				if (err != nil) != tt.wantErr {
					t.Errorf("Utf16OffsetToBytes() error = %v, wantErr %v", err, tt.wantErr)
					return // Stop if error status differs
				}
				// Check specific error type if expecting an error
				if tt.wantErr && tt.wantErrType != nil {
					if !errors.Is(err, tt.wantErrType) {
						t.Errorf("Utf16OffsetToBytes() error type = %T, want error wrapping %T", err, tt.wantErrType)
					}
				}
				// Check byte offset only if error status matches expectation
				// Note: If wantErr is true, Utf16OffsetToBytes might return a clamped offset,
				// so we check against wantByteOffset even on error cases where clamping occurs.
				if gotByteOffset != tt.wantByteOffset {
					t.Errorf("Utf16OffsetToBytes() gotByteOffset = %d, want %d", gotByteOffset, tt.wantByteOffset)
				}
			})
		}
	})

	// --- Test LspPositionToBytePosition ---
	t.Run("TestLspPositionToBytePosition", func(t *testing.T) {
		content := `line one
two é 😂
three €
` // includes trailing newline

		tests := []struct {
			name        string
			content     []byte
			lspPos      LSPPosition
			wantLine    int // 1-based Go line
			wantCol     int // 1-based Go column (bytes)
			wantByteOff int // 0-based byte offset
			wantErr     bool
			wantErrType error  // Specific error type to check (e.g., ErrPositionConversion)
			wantWarnLog string // Substring expected in log output (for clamping warnings)
		}{
			// Basic cases
			{"Start of file", []byte(content), LSPPosition{Line: 0, Character: 0}, 1, 1, 0, false, nil, ""},
			{"Middle line 1", []byte(content), LSPPosition{Line: 0, Character: 5}, 1, 6, 5, false, nil, ""}, // After ' '
			{"End line 1", []byte(content), LSPPosition{Line: 0, Character: 8}, 1, 9, 8, false, nil, ""},    // End of 'one'

			// Line 2 with unicode (line starts at byte offset 9)
			{"Start line 2", []byte(content), LSPPosition{Line: 1, Character: 0}, 2, 1, 9, false, nil, ""},                 // Start of 'two'
			{"Middle line 2 (before é)", []byte(content), LSPPosition{Line: 1, Character: 4}, 2, 5, 13, false, nil, ""},    // After ' ' before 'é' (t=1,w=1,o=1, =1 => 4 chars, 4 bytes) -> Go Col 4+1=5, Offset 9+4=13
			{"Middle line 2 (after é)", []byte(content), LSPPosition{Line: 1, Character: 5}, 2, 7, 15, false, nil, ""},     // After 'é' (1 utf16 unit, 2 bytes) -> Go Col 5+2=7, Offset 13+2=15
			{"Middle line 2 (after space)", []byte(content), LSPPosition{Line: 1, Character: 6}, 2, 8, 16, false, nil, ""}, // After ' ' before 😂 (1 utf16, 1 byte) -> Go Col 7+1=8, Offset 15+1=16
			{"Middle line 2 (within 😂)", []byte(content), LSPPosition{Line: 1, Character: 7}, 2, 8, 16, false, nil, ""},    // Within 😂 surrogate pair (byte offset remains at start of rune) -> Go Col 8, Offset 16
			{"Middle line 2 (after 😂)", []byte(content), LSPPosition{Line: 1, Character: 8}, 2, 12, 20, false, nil, ""},    // After 😂 (2 utf16 units, 4 bytes) -> Go Col 8+4=12, Offset 16+4=20
			{"End line 2", []byte(content), LSPPosition{Line: 1, Character: 8}, 2, 12, 20, false, nil, ""},                 // End of 😂 (same as after)

			// Line 3 with unicode (line starts at byte offset 21)
			{"Start line 3", []byte(content), LSPPosition{Line: 2, Character: 0}, 3, 1, 21, false, nil, ""},                // Start of 'three'
			{"Middle line 3 (after space)", []byte(content), LSPPosition{Line: 2, Character: 6}, 3, 7, 27, false, nil, ""}, // After ' ' before '€' (t=1,h=1,r=1,e=1,e=1, =1 => 6 chars, 6 bytes) -> Go Col 6+1=7, Offset 21+6=27
			{"Middle line 3 (after €)", []byte(content), LSPPosition{Line: 2, Character: 7}, 3, 10, 30, false, nil, ""},    // After '€' (1 utf16 unit, 3 bytes) -> Go Col 7+3=10, Offset 27+3=30
			{"End line 3", []byte(content), LSPPosition{Line: 2, Character: 7}, 3, 10, 30, false, nil, ""},                 // End of '€' (same as after)

			// Line 4 (empty line after content due to trailing newline, starts at offset 31)
			{"Start empty line 4", []byte(content), LSPPosition{Line: 3, Character: 0}, 4, 1, 31, false, nil, ""}, // Start of the line after the last newline

			// Edge cases and errors
			{"Nil content", nil, LSPPosition{Line: 0, Character: 0}, 0, 0, -1, true, ErrPositionConversion, ""},
			{"Empty content", []byte(""), LSPPosition{Line: 0, Character: 0}, 1, 1, 0, false, nil, ""},                                // Start of empty file is valid
			{"Empty content invalid char", []byte(""), LSPPosition{Line: 0, Character: 1}, 0, 0, -1, true, ErrPositionOutOfRange, ""}, // Char > 0 on empty line -> Utf16OffsetToBytes returns ErrPositionOutOfRange
			{"Empty content invalid line", []byte(""), LSPPosition{Line: 1, Character: 0}, 0, 0, -1, true, ErrPositionOutOfRange, ""}, // Line > 0 on empty file

			{"Negative line", []byte(content), LSPPosition{Line: uint32(int32(-1)), Character: 0}, 0, 0, -1, true, ErrInvalidPositionInput, ""}, // Simulate negative line via cast
			{"Negative char", []byte(content), LSPPosition{Line: 0, Character: uint32(int32(-1))}, 0, 0, -1, true, ErrInvalidPositionInput, ""}, // Simulate negative char via cast

			{"Line past end", []byte(content), LSPPosition{Line: 4, Character: 0}, 0, 0, -1, true, ErrPositionOutOfRange, ""},              // Line 4 doesn't exist (only 0-3)
			{"Char past end line 1 (Clamps)", []byte(content), LSPPosition{Line: 0, Character: 10}, 1, 9, 8, false, nil, "Clamping"},       // Clamp char past 'one' (utf16 offset 10 > len 8) -> byte offset 8 (end), col 9
			{"Char past end line 2 (Clamps)", []byte(content), LSPPosition{Line: 1, Character: 9}, 2, 12, 20, false, nil, "Clamping"},      // Clamp char past '😂' (utf16 offset 9 > len 8) -> byte offset 20 (end), col 12
			{"Char past end line 3 (Clamps)", []byte(content), LSPPosition{Line: 2, Character: 8}, 3, 10, 30, false, nil, "Clamping"},      // Clamp char past '€' (utf16 offset 8 > len 7) -> byte offset 30 (end), col 10
			{"Char past end empty line 4", []byte(content), LSPPosition{Line: 3, Character: 1}, 0, 0, -1, true, ErrPositionOutOfRange, ""}, // Char > 0 on empty line
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				// Capture log output for checking warnings
				var logBuf bytes.Buffer
				// Use a test-specific logger temporarily
				testLogger := log.New(&logBuf, "TEST: ", log.Lshortfile)
				// Swap std logger, restore after test
				oldLogger := log.Default() // Get current default logger
				log.SetOutput(testLogger.Writer())
				log.SetFlags(testLogger.Flags())
				log.SetPrefix(testLogger.Prefix())
				t.Cleanup(func() {
					// Restore previous logger settings
					log.SetOutput(oldLogger.Writer())
					log.SetFlags(oldLogger.Flags())
					log.SetPrefix(oldLogger.Prefix())
				})

				// Call the function under test
				gotLine, gotCol, gotByteOff, err := LspPositionToBytePosition(tt.content, tt.lspPos)

				// Check error status
				if (err != nil) != tt.wantErr {
					t.Errorf("LspPositionToBytePosition() error = %v, wantErr %v", err, tt.wantErr)
					return // Stop checking other values if error status is wrong
				}

				// Check specific error type if specified
				if tt.wantErr && tt.wantErrType != nil {
					if !errors.Is(err, tt.wantErrType) {
						t.Errorf("LspPositionToBytePosition() error type = %T, want error wrapping %T", err, tt.wantErrType)
					}
				}

				// Check returned values only if error status matches expectation
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

				// Check log output for expected warnings (e.g., clamping)
				logOutput := logBuf.String()
				if tt.wantWarnLog != "" && !strings.Contains(logOutput, tt.wantWarnLog) {
					t.Errorf("LspPositionToBytePosition() log output missing expected warning containing %q. Got:\n%s", tt.wantWarnLog, logOutput)
				}
				if tt.wantWarnLog == "" && strings.Contains(logOutput, "Warning:") {
					// Only fail if an *unexpected* warning occurred
					t.Errorf("LspPositionToBytePosition() logged unexpected warning. Got:\n%s", logOutput)
				}
			})
		}
	})
}
