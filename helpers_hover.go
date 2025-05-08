// deepcomplete/helpers_hover.go
// Contains helper functions specifically for generating hover information.
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

// formatObjectForHover creates a Markdown string for hover info based on IdentifierInfo.
// It includes the object's definition signature and its associated documentation comment, if found.
func formatObjectForHover(identInfo *IdentifierInfo, logger *slog.Logger) string {
	if identInfo == nil || identInfo.Object == nil {
		logger.Debug("formatObjectForHover called with nil identInfo or object")
		return ""
	}
	obj := identInfo.Object // Extract object from IdentifierInfo

	if logger == nil {
		logger = slog.Default() // Fallback if logger is nil
	}
	hoverLogger := logger.With("object_name", obj.Name(), "object_type", fmt.Sprintf("%T", obj))

	var hoverText strings.Builder

	// --- 1. Format Definition ---
	var qualifier types.Qualifier
	// Use package info from IdentifierInfo if available
	if identInfo.Pkg != nil && identInfo.Pkg.Types != nil {
		qualifier = types.RelativeTo(identInfo.Pkg.Types)
		hoverLogger.Debug("Using package qualifier for hover formatting")
	} else {
		qualifier = func(other *types.Package) string {
			if other != nil {
				return other.Name() // Use name as fallback
			}
			return ""
		}
		hoverLogger.Warn("Target package or types missing in IdentifierInfo, using fallback qualifier for hover.")
	}

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

	// Use definition node and fileset from IdentifierInfo
	if identInfo.DefNode != nil && identInfo.FileSet != nil {
		defNodePosStr := getPosString(identInfo.FileSet, identInfo.DefNode.Pos())
		hoverLogger.Debug("Attempting to find doc comment on definition node", "def_node_type", fmt.Sprintf("%T", identInfo.DefNode), "def_node_pos", defNodePosStr)

		switch n := identInfo.DefNode.(type) {
		case *ast.FuncDecl:
			commentGroup = n.Doc
		case *ast.GenDecl:
			foundSpecDoc := false
			for _, spec := range n.Specs {
				var specDoc *ast.CommentGroup
				var specPos token.Pos = token.NoPos
				match := false
				switch s := spec.(type) {
				case *ast.ValueSpec:
					for _, name := range s.Names {
						if name != nil && name.Pos() == obj.Pos() {
							specDoc = s.Doc
							specPos = s.Pos()
							match = true
							break
						}
					}
				case *ast.TypeSpec:
					if s.Name != nil && s.Name.Pos() == obj.Pos() {
						specDoc = s.Doc
						specPos = s.Pos()
						match = true
					}
				}
				if match {
					commentGroup = specDoc
					specPosStr := getPosString(identInfo.FileSet, specPos)
					if commentGroup == nil {
						commentGroup = n.Doc
						hoverLogger.Debug("Using GenDecl doc as fallback", "spec_type", fmt.Sprintf("%T", spec), "spec_pos", specPosStr)
					} else {
						hoverLogger.Debug("Found doc comment on specific Spec node", "spec_type", fmt.Sprintf("%T", spec), "spec_pos", specPosStr)
					}
					foundSpecDoc = true
					break
				}
			}
			if !foundSpecDoc {
				commentGroup = n.Doc
				genDeclPosStr := getPosString(identInfo.FileSet, n.Pos())
				hoverLogger.Debug("No matching Spec found in GenDecl, using GenDecl doc", "gen_decl_pos", genDeclPosStr)
			}
		case *ast.TypeSpec:
			commentGroup = n.Doc
		case *ast.Field:
			commentGroup = n.Doc
		case *ast.ValueSpec:
			commentGroup = n.Doc
		case *ast.AssignStmt:
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
		hoverLogger.Debug("Defining node (DefNode) or FileSet not found in IdentifierInfo, cannot get doc comment.")
	}

	if commentGroup != nil && len(commentGroup.List) > 0 {
		var doc strings.Builder
		for _, c := range commentGroup.List {
			if c != nil {
				text := strings.TrimSpace(strings.TrimPrefix(c.Text, "//"))
				text = strings.TrimSpace(strings.TrimPrefix(text, "/*"))
				text = strings.TrimSpace(strings.TrimSuffix(text, "*/"))
				if doc.Len() > 0 {
					doc.WriteString("\n")
				}
				doc.WriteString(text)
			}
		}
		docComment = doc.String()
		hoverLogger.Debug("Found and formatted doc comment", "comment_length", len(docComment))
	} else if identInfo.DefNode != nil {
		hoverLogger.Debug("No doc comment found on definition node")
	}

	// --- 3. Combine Definition and Documentation ---
	if docComment != "" {
		if hoverText.Len() > 0 {
			hoverText.WriteString("\n\n---\n\n") // Separator
		}
		hoverText.WriteString(docComment)
	}

	finalContent := hoverText.String()
	if strings.TrimSpace(finalContent) == "```go\n```" || strings.TrimSpace(finalContent) == "" {
		hoverLogger.Debug("No hover content generated (empty definition and no docs).")
		return ""
	}

	hoverLogger.Debug("Formatted hover content generated", "content_length", len(finalContent))
	return finalContent
}
