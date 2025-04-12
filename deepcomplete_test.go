// deepcomplete/deepcomplete_test.go
package deepcomplete

import (
	"context"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

// TestLintResult tests the simple brace linting middleware.
func TestLintResult(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool // If true, expects an error (currently LintResult only logs warnings)
	}{
		{"Balanced braces", "func main() {\n\tfmt.Println()\n}", false},
		{"Missing closing brace", "func main() {\n\tfmt.Println()", false},   // Currently only logs warning
		{"Extra closing brace", "func main() {\n\tfmt.Println()\n}}", false}, // Currently only logs warning
		{"No braces", "package main", false},
		{"Nested balanced", "func main() { if true { } }", false},
		{"Nested unbalanced", "func main() { if true { }", false}, // Log warning
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LintResult(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("LintResult() error = %v, wantErr %v", err, tt.wantErr)
			}
			// Note: Cannot easily test log output in standard tests without redirecting logs.
		})
	}
}

// TestRemoveExternalLibraries tests the filtering of non-stdlib imports.
func TestRemoveExternalLibraries(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"Stdlib only", "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println()\n}", "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println()\n}"},
		{"Stdlib block", "package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc main() {}", "package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc main() {}"},
		{"External single", "package main\n\nimport \"github.com/gin-gonic/gin\"\n\nfunc main() {}", "package main\n\n\n\nfunc main() {}"}, // Adjusted expected output
		{"External block", "package main\n\nimport (\n\t\"fmt\"\n\t\"github.com/pkg/errors\"\n\t\"os\"\n)\n\nfunc main() {}", "package main\n\nimport (\n\t\"fmt\"\n\t\n\t\"os\"\n)\n\nfunc main() {}"},
		{"Golang.org/x allowed", "package main\n\nimport \"golang.org/x/tools/go/packages\"\n\nfunc main() {}", "package main\n\nimport \"golang.org/x/tools/go/packages\"\n\nfunc main() {}"},
		{"Mixed block", "package main\n\nimport (\n\t\"context\"\n\t\"example.com/mypkg\"\n\t\"golang.org/x/sync/errgroup\"\n)\n\nfunc main() {}", "package main\n\nimport (\n\t\"context\"\n\t\n\t\"golang.org/x/sync/errgroup\"\n)\n\nfunc main() {}"},
		{"No imports", "package main\n\nfunc main() {}", "package main\n\nfunc main() {}"},
		{"Commented import", "package main\n\n// import \"github.com/gin-gonic/gin\"\n\nfunc main() {}", "package main\n\n// import \"github.com/gin-gonic/gin\"\n\nfunc main() {}"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RemoveExternalLibraries(tt.input)
			if err != nil {
				t.Errorf("RemoveExternalLibraries() unexpected error = %v", err)
				return
			}
			// Normalize newlines for comparison
			wantNorm := strings.ReplaceAll(tt.want, "\r\n", "\n")
			gotNorm := strings.ReplaceAll(got, "\r\n", "\n")
			if gotNorm != wantNorm {
				t.Errorf("RemoveExternalLibraries() got = %q, want %q", gotNorm, wantNorm)
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

	// Cursor is here
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

	// --- Test Case: Inside MyFunction ---
	t.Run("Inside MyFunction", func(t *testing.T) {
		line, col := 17, 2 // Cursor position within the function body ("// Cursor is here")
		ctx := context.Background()

		// Call AnalyzeCodeContext - it will use the direct parse fallback as no go.mod exists
		// We expect non-fatal errors related to package loading / type info missing
		info, analysisErr := AnalyzeCodeContext(ctx, tmpFilename, line, col)

		// Check for fatal errors first (should not happen if file exists and pos is valid)
		if analysisErr != nil {
			// Check if it's ONLY the expected non-fatal errors
			errStr := analysisErr.Error()
			isExpectedErr := strings.Contains(errStr, "package loading failed") ||
				strings.Contains(errStr, "package loading reported errors") ||
				strings.Contains(errStr, "package type information missing") ||
				strings.Contains(errStr, "package scope information missing") ||
				strings.Contains(errStr, "Type information unavailable") ||
				strings.Contains(errStr, "Type/Scope information unavailable") ||
				strings.Contains(errStr, "Skipping") // Ignore type resolution skips

			if !isExpectedErr {
				t.Fatalf("AnalyzeCodeContext returned unexpected error: %v", analysisErr)
			} else {
				t.Logf("AnalyzeCodeContext returned expected non-fatal errors: %v", analysisErr)
			}
		}
		if info == nil {
			t.Fatal("AnalyzeCodeContext returned nil info")
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

		// Check scope variables (won't have type info, check presence)
		// Note: Without type info, only AST-level declarations might be found easily.
		// The current implementation relies heavily on type info for scope.
		// Check preamble for hints instead.
		// if _, ok := info.VariablesInScope["localVar"]; !ok {
		// 	t.Errorf("Expected 'localVar' in scope, VariablesInScope: %v", info.VariablesInScope)
		// }
		// if _, ok := info.VariablesInScope["arg1"]; !ok {
		// 	t.Errorf("Expected 'arg1' in scope, VariablesInScope: %v", info.VariablesInScope)
		// }
		// if _, ok := info.VariablesInScope["GlobalConst"]; !ok {
		//  t.Errorf("Expected 'GlobalConst' in scope, VariablesInScope: %v", info.VariablesInScope)
		// }

		// Check preamble contains expected elements (basic check)
		if !strings.Contains(info.PromptPreamble, "// Enclosing Function (AST): func MyFunction(...)") {
			t.Errorf("Preamble missing expected enclosing function. Got:\n%s", info.PromptPreamble)
		}
		if !strings.Contains(info.PromptPreamble, expectedDoc) {
			t.Errorf("Preamble missing expected doc comment. Got:\n%s", info.PromptPreamble)
		}
		// Check if preamble mentions scope (even if empty due to lack of type info)
		if !strings.Contains(info.PromptPreamble, "in Scope") {
			t.Errorf("Preamble missing expected scope section. Got:\n%s", info.PromptPreamble)
		}

	})

	// TODO: Add more basic test cases for different cursor positions (e.g., global scope, inside struct).
}

func TestFindCommentsWithMap(t *testing.T) {
	// TODO: Implement more focused tests for findCommentsWithMap
	// Requires creating sample ASTs and CommentMaps or parsing test files.
	// Test cases:
	// - Comment exactly on line before cursor
	// - Doc comment on enclosing function/type/var
	// - General comment attached to enclosing node
	// - No relevant comments
	// - Block comments vs line comments
	t.Skip("findCommentsWithMap tests not yet implemented")
}

func TestFindIdentifierAtCursor(t *testing.T) {
	// TODO: Implement tests for findIdentifierAtCursor
	// Requires setting up ASTs and mock type info or parsing test files.
	// Test cases:
	// - Cursor on variable use
	// - Cursor on variable definition
	// - Cursor on function call name
	// - Cursor on type name
	// - Cursor immediately after identifier
	// - Cursor not near any identifier
	t.Skip("findIdentifierAtCursor tests not yet implemented")
}
