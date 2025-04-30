// deepcomplete/helpers_loader.go
package deepcomplete

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"log/slog"
	"path/filepath"

	"golang.org/x/tools/go/packages"
)

// loadPackageAndFile encapsulates the logic for loading package information using go/packages.
// It attempts to load the package containing the specified file and extracts relevant
// AST, token file, package info, diagnostics, and any loading errors encountered.
// Returns: targetPkg, targetFileAST, targetFile, diagnostics, loadErrors
func loadPackageAndFile(ctx context.Context, absFilename string, fset *token.FileSet, logger *slog.Logger) (*packages.Package, *ast.File, *token.File, []Diagnostic, []error) {
	var loadErrors []error
	var diagnostics []Diagnostic // Collect diagnostics during loading
	dir := filepath.Dir(absFilename)
	logger = logger.With("loadDir", dir)

	// Configuration for loading packages. NeedSyntax is crucial for AST.
	// NeedTypesInfo provides access to type checker results (definitions, uses, types, scopes).
	loadCfg := &packages.Config{
		Context: ctx,
		Dir:     dir,
		Fset:    fset,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedTypesSizes |
			packages.NeedSyntax | packages.NeedTypesInfo,
		// NOTE: packages.NeedErrors is implicitly included by NeedTypes / NeedSyntax etc.
		Tests: false, // Don't load test files initially
		Logf:  func(format string, args ...interface{}) { logger.Debug(fmt.Sprintf(format, args...)) },
	}

	logger.Debug("Calling packages.Load")
	pkgs, err := packages.Load(loadCfg, fmt.Sprintf("file=%s", absFilename))
	if err != nil {
		// Critical error during packages.Load itself
		loadErrors = append(loadErrors, fmt.Errorf("packages.Load failed critically: %w", err))
		logger.Error("packages.Load failed critically", "error", err)
		return nil, nil, nil, diagnostics, loadErrors // Return collected diagnostics even on critical failure
	}

	if len(pkgs) == 0 {
		// packages.Load succeeded but returned no packages for the file pattern
		loadErrors = append(loadErrors, errors.New("packages.Load returned no packages"))
		logger.Warn("packages.Load returned no packages.")
		return nil, nil, nil, diagnostics, loadErrors
	}

	var targetPkg *packages.Package
	var targetFileAST *ast.File
	var targetFile *token.File

	// Collect all package-level errors first and convert to diagnostics
	for _, p := range pkgs {
		for i := range p.Errors {
			pkgErr := p.Errors[i] // Make a copy to avoid loop variable issues
			loadErrors = append(loadErrors, &pkgErr)
			logger.Warn("Package loading error encountered", "package", p.PkgPath, "error", pkgErr)
			// Convert packages.Error to our internal Diagnostic format
			diag := packagesErrorToDiagnostic(pkgErr, fset, logger) // Assumes helpers_diagnostics.go exists
			if diag != nil {
				diagnostics = append(diagnostics, *diag)
			}
		}
	}

	// Find the specific package and AST file corresponding to absFilename
	for _, p := range pkgs {
		if p == nil {
			continue
		}
		logger.Debug("Checking package", "pkg_id", p.ID, "pkg_path", p.PkgPath, "num_files", len(p.CompiledGoFiles), "num_syntax", len(p.Syntax))
		for i, astFile := range p.Syntax {
			if astFile == nil {
				logger.Warn("Nil AST file encountered in package syntax", "package", p.PkgPath, "index", i)
				continue
			}
			// Get the position of the AST file's start
			filePos := fset.Position(astFile.Pos())
			if !filePos.IsValid() {
				// This shouldn't happen if the AST node is valid
				logger.Warn("AST file has invalid position", "package", p.PkgPath, "index", i)
				continue
			}
			// Compare absolute paths for robustness
			astFilePath, _ := filepath.Abs(filePos.Filename)
			logger.Debug("Comparing file paths", "ast_file_path", astFilePath, "target_file_path", absFilename)
			if astFilePath == absFilename {
				targetPkg = p
				targetFileAST = astFile
				// Get the token.File using the valid position from the FileSet
				targetFile = fset.File(astFile.Pos())
				if targetFile == nil {
					// This should ideally not happen if astFile.Pos() is valid and fset is correct
					fileNotFoundErr := fmt.Errorf("failed to get token.File for AST file %s from FileSet", absFilename)
					loadErrors = append(loadErrors, fileNotFoundErr)
					logger.Error("Could not get token.File from FileSet even with valid AST position", "filename", absFilename)
				}
				logger.Debug("Found target file in package", "package", p.PkgPath)
				goto foundTarget // Exit loops once target is found
			}
		}
	}

foundTarget:
	// Handle case where the specific file wasn't found in the loaded packages' syntax trees
	if targetPkg == nil {
		err := fmt.Errorf("target file %s not found in loaded packages syntax trees", absFilename)
		loadErrors = append(loadErrors, err)
		logger.Warn(err.Error())
		// Attempt to return any diagnostics found even if the target wasn't located
		return nil, nil, nil, diagnostics, loadErrors
	}

	// Final checks on the found package (if any)
	// Check if critical type information is missing, which indicates deeper issues
	if targetPkg.TypesInfo == nil {
		loadErrors = append(loadErrors, fmt.Errorf("type info (TypesInfo) is nil for target package %s", targetPkg.PkgPath))
		logger.Warn("TypesInfo is nil for target package", "package", targetPkg.PkgPath)
	}
	if targetPkg.Types == nil {
		loadErrors = append(loadErrors, fmt.Errorf("types (Types) is nil for target package %s", targetPkg.PkgPath))
		logger.Warn("Types is nil for target package", "package", targetPkg.PkgPath)
	}

	return targetPkg, targetFileAST, targetFile, diagnostics, loadErrors
}
