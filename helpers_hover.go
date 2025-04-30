// deepcomplete/helpers_hover.go
package deepcomplete

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log/slog"
	"strings"
)

// ============================================================================
// Hover Formatting Helper
// ============================================================================

// formatObjectForHover creates a Markdown string for hover info based on the resolved types.Object.
// It includes the object's definition signature and its associated documentation comment, if found.
func formatObjectForHover(obj types.Object, info *AstContextInfo, logger *slog.Logger) string {
	if obj == nil {
		logger.Debug("formatObjectForHover called with nil object")
		return ""
	}
	hoverLogger := logger.With("object_name", obj.Name(), "object_type", fmt.Sprintf("%T", obj))

	var hoverText strings.Builder

	// --- 1. Format Definition ---
	var qualifier types.Qualifier
	if info.TargetPackage != nil && info.TargetPackage.Types != nil {
		qualifier = types.RelativeTo(info.TargetPackage.Types)
		hoverLogger.Debug("Using package qualifier for hover formatting")
	} else {
		// Fallback qualifier if package context is unavailable
		qualifier = func(other *types.Package) string {
			if other != nil {
				return other.Name() // Use package name if possible
			}
			return "" // Empty string otherwise
		}
		hoverLogger.Warn("Target package or types missing, using fallback qualifier for hover.")
	}

	// Use types.ObjectString for a concise definition string (e.g., "var name type", "func name(...) ...")
	definition := types.ObjectString(obj, qualifier)
	if definition != "" {
		hoverText.WriteString("```go\n") // Start Go code block
		hoverText.WriteString(definition)
		hoverText.WriteString("\n```") // End code block
		hoverLogger.Debug("Formatted object definition", "definition", definition)
	} else {
		hoverLogger.Warn("Could not format object definition string")
	}

	// --- 2. Find and Format Documentation ---
	docComment := ""
	var commentGroup *ast.CommentGroup

	// Attempt to get comment from the definition node found earlier during analysis
	if info.IdentifierDefNode != nil {
		hoverLogger.Debug("Attempting to find doc comment on definition node", "def_node_type", fmt.Sprintf("%T", info.IdentifierDefNode), "def_node_pos", getPosString(info.TargetFileSet, info.IdentifierDefNode.Pos())) // Assumes helpers_diagnostics.go exists
		// Extract the comment group based on the type of the definition node
		switch n := info.IdentifierDefNode.(type) {
		case *ast.FuncDecl:
			commentGroup = n.Doc
		case *ast.GenDecl: // Handles var, const, type blocks
			// For GenDecl, find the specific Spec that matches the object's position
			foundSpecDoc := false
			for _, spec := range n.Specs {
				var specDoc *ast.CommentGroup
				var specPos token.Pos = token.NoPos
				match := false
				switch s := spec.(type) {
				case *ast.ValueSpec: // var, const
					for _, name := range s.Names {
						if name != nil && name.Pos() == obj.Pos() {
							specDoc = s.Doc
							specPos = s.Pos()
							match = true
							break
						}
					}
				case *ast.TypeSpec: // type
					if s.Name != nil && s.Name.Pos() == obj.Pos() {
						specDoc = s.Doc
						specPos = s.Pos()
						match = true
					}
				}
				// If the specific spec matched the object's definition position
				if match {
					commentGroup = specDoc
					// If the spec itself has no doc, fall back to the GenDecl's doc
					if commentGroup == nil {
						commentGroup = n.Doc
						hoverLogger.Debug("Using GenDecl doc as fallback", "spec_type", fmt.Sprintf("%T", spec), "spec_pos", getPosString(info.TargetFileSet, specPos))
					} else {
						hoverLogger.Debug("Found doc comment on specific Spec node", "spec_type", fmt.Sprintf("%T", spec), "spec_pos", getPosString(info.TargetFileSet, specPos))
					}
					foundSpecDoc = true
					break // Exit spec loop once the matching spec is found
				}
			}
			// If no specific spec matched (e.g., cursor on `var` keyword itself?), use the GenDecl doc.
			if !foundSpecDoc {
				commentGroup = n.Doc
				hoverLogger.Debug("No matching Spec found in GenDecl, using GenDecl doc", "gen_decl_pos", getPosString(info.TargetFileSet, n.Pos()))
			}
		case *ast.TypeSpec: // Handles individual type specs outside GenDecl (less common)
			commentGroup = n.Doc
		case *ast.Field: // Handles struct fields, interface methods, func params/results
			commentGroup = n.Doc
		case *ast.ValueSpec: // Handles individual var/const specs outside GenDecl
			commentGroup = n.Doc
		case *ast.AssignStmt: // Handle short variable declarations (var := value)
			// Short variable declarations (`:=`) don't have associated doc comments in the AST.
			if n.Tok == token.DEFINE {
				for _, lhsExpr := range n.Lhs {
					if ident, ok := lhsExpr.(*ast.Ident); ok && ident.Pos() == obj.Pos() {
						hoverLogger.Debug("Hover object is a short variable declaration; doc comment lookup not supported for this case.")
						break
					}
				}
			}
		default:
			hoverLogger.Debug("Hover documentation lookup: Unhandled definition node type", "type", fmt.Sprintf("%T", n))
		}
	} else {
		hoverLogger.Debug("Defining node (IdentifierDefNode) not found in AstContextInfo, cannot get doc comment.")
	}

	// Format the comment group text if found
	if commentGroup != nil && len(commentGroup.List) > 0 {
		var doc strings.Builder
		for _, c := range commentGroup.List {
			if c != nil {
				// Basic cleaning: remove comment markers, trim space
				text := strings.TrimSpace(strings.TrimPrefix(c.Text, "//"))
				text = strings.TrimSpace(strings.TrimPrefix(text, "/*"))
				text = strings.TrimSpace(strings.TrimSuffix(text, "*/"))
				if doc.Len() > 0 {
					doc.WriteString("\n") // Add newline between comment lines for readability
				}
				doc.WriteString(text)
			}
		}
		docComment = doc.String()
		hoverLogger.Debug("Found and formatted doc comment", "comment_length", len(docComment))
	} else if info.IdentifierDefNode != nil {
		// Log only if we expected to find a node but it had no comment
		hoverLogger.Debug("No doc comment found on definition node")
	}

	// --- 3. Combine Definition and Documentation ---
	if docComment != "" {
		if hoverText.Len() > 0 {
			// Add a separator if both definition and documentation exist
			hoverText.WriteString("\n\n---\n\n")
		}
		hoverText.WriteString(docComment)
	}

	finalContent := hoverText.String()
	// Avoid returning just an empty code block or whitespace
	if strings.TrimSpace(finalContent) == "```go\n```" || strings.TrimSpace(finalContent) == "" {
		hoverLogger.Debug("No hover content generated (empty definition and no docs).")
		return ""
	}

	hoverLogger.Debug("Formatted hover content generated", "content_length", len(finalContent))
	return finalContent
}
