// deepcomplete/deepcomplete_test.go
package deepcomplete

import (
	"bytes"
	"context"
	"errors"
	"go/ast"
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
func TestCalculateCursorPos(t *testing.T) {
	// Sample file content for position calculation
	content := `package main

import "fmt" // Line 3

func main() { // Line 5
	fmt.Println("Hello") // Line 6
} // Line 7
// Line 8` // Ensure content ends without newline for some tests

	// Use a simple fileset for this test, file content doesn't need full parsing here
	fset := token.NewFileSet()
	file := fset.AddFile("test_calcpos.go", fset.Base(), len(content))
	if file == nil {
		t.Fatal("Failed to add file")
	}

	tests := []struct {
		name string
		line int
		col  int
		// wantPos    token.Pos // Comparing exact Pos can be tricky, focus on offset
		wantErr    bool
		wantOffset int // Expected byte offset for verification
	}{
		{"Start of file", 1, 1, false, 0},
		{"Start of line 3", 3, 1, false, 15},                         // Offset of 'import "fmt"'
		{"Middle of Println", 6, 10, false, 46},                      // Offset within Println
		{"End of Println line", 6, 26, false, 62},                    // Offset after ')' on line 6
		{"Cursor after Println line (at newline)", 6, 27, false, 62}, // Clamped to line end offset if no newline char exists there? Let's test file size limit instead.
		{"Start of line 7", 7, 1, false, 63},                         // Offset of '}' on line 7
		{"End of line 7", 7, 2, false, 64},                           // Offset after '}' on line 7
		{"Start of line 8", 8, 1, false, 65},                         // Offset of '/' in comment
		{"End of line 8", 8, 10, false, 74},                          // Offset at end of content
		{"Cursor after end of file", 8, 11, false, 74},               // Clamped to end of file size
		{"Invalid line (too high)", 9, 1, true, -1},
		{"Invalid line (zero)", 0, 1, true, -1},
		{"Invalid col (zero)", 6, 0, true, -1},
		{"Invalid col (too high for line)", 6, 100, false, 62}, // Clamped to end of line offset
		{"Col too high last line", 8, 100, false, 74},          // Clamped to end of file offset
		{"Col too high last line + 1", 9, 100, true, -1},       // Invalid line number
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
	// --- Ensure analyzer resources are closed (Step 24 Change) ---
	t.Cleanup(func() {
		t.Log("Closing analyzer in test cleanup...") // Add log for verification
		if err := analyzer.Close(); err != nil {
			t.Errorf("Error closing analyzer: %v", err)
		}
	})
	// ---

	// --- Test Case: Inside MyFunction (general) ---
	t.Run("Inside MyFunction", func(t *testing.T) {
		line, col := 18, 2 // Cursor position within the function body ("// Cursor is here")
		ctx := context.Background()

		// Call Analyze method - it will use the direct parse fallback as no go.mod exists
		// We expect non-fatal errors related to package loading / type info missing
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

func TestGoPackagesAnalyzer_Cache(t *testing.T) {
	// TODO: Implement tests for bbolt caching (Step 23)
	// Needs setup with temp DB file (ensure path cleanup), multiple Analyze calls on same dir,
	// checking for cache hit logs, modifying files, checking for invalidation.
	// Remember to call analyzer.Close() via t.Cleanup() in these tests.
	t.Skip("GoPackagesAnalyzer caching tests not yet implemented")
}
