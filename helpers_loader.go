// deepcomplete/helpers_loader.go
// Contains helper functions specifically for loading packages using go/packages.
// Cycle 3: Added context propagation, explicit logger passing, refined error handling.
package deepcomplete

import (
	"context" // Added for context passing
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
func loadPackageAndFile(
	ctx context.Context, // Pass context
	absFilename string,
	fset *token.FileSet,
	logger *slog.Logger,
) (*packages.Package, *ast.File, *token.File, []Diagnostic, []error) {
	var loadErrors []error
	var diagnostics []Diagnostic // Collect diagnostics during loading
	dir := filepath.Dir(absFilename)
	if logger == nil {
		logger = slog.Default() // Fallback if logger is nil
	}
	logger = logger.With("loadDir", dir)

	// Configuration for loading packages.
	loadCfg := &packages.Config{
		Context: ctx, // Use passed context
		Dir:     dir,
		Fset:    fset,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedTypesSizes |
			packages.NeedSyntax | packages.NeedTypesInfo,
		Tests: false, // Don't load test files initially
		// Pass logger to packages.Load for its internal logging
		Logf: func(format string, args ...interface{}) { logger.Debug(fmt.Sprintf(format, args...)) },
	}

	logger.Debug("Calling packages.Load")
	pkgs, err := packages.Load(loadCfg, fmt.Sprintf("file=%s", absFilename))
	if err != nil {
		// Critical error during packages.Load itself
		loadErr := fmt.Errorf("packages.Load failed critically: %w", err)
		loadErrors = append(loadErrors, loadErr)
		logger.Error("packages.Load failed critically", "error", err)
		// Check for context cancellation specifically
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			// Don't add a generic diagnostic if it was a context error
		} else {
			// Add a general diagnostic for critical load failure
			// Use a placeholder range at the start of the file
			diag := Diagnostic{
				Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
				Severity: SeverityError,
				Source:   "deepcomplete-loader",
				Message:  fmt.Sprintf("Critical error loading package: %v", err),
			}
			diagnostics = append(diagnostics, diag)
		}
		return nil, nil, nil, diagnostics, loadErrors // Return collected diagnostics even on critical failure
	}

	if len(pkgs) == 0 {
		// packages.Load succeeded but returned no packages
		loadErr := errors.New("packages.Load returned no packages")
		loadErrors = append(loadErrors, loadErr)
		logger.Warn(loadErr.Error())
		// Add a diagnostic indicating no package was found
		diag := Diagnostic{
			Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
			Severity: SeverityWarning, // Warning, as it might not be a fatal error for all operations
			Source:   "deepcomplete-loader",
			Message:  fmt.Sprintf("Could not load package information for file: %s", absFilename),
		}
		diagnostics = append(diagnostics, diag)
		return nil, nil, nil, diagnostics, loadErrors
	}

	var targetPkg *packages.Package
	var targetFileAST *ast.File
	var targetFile *token.File

	// Collect all package-level errors first and convert to diagnostics
	for _, p := range pkgs {
		if p == nil {
			continue
		} // Defensive check
		for i := range p.Errors {
			pkgErr := p.Errors[i] // Make a copy
			loadErrors = append(loadErrors, &pkgErr)
			logger.Warn("Package loading error encountered", "package", p.PkgPath, "error", pkgErr.Error())
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
					fileNotFoundErr := fmt.Errorf("failed to get token.File for AST file %s from FileSet", absFilename)
					loadErrors = append(loadErrors, fileNotFoundErr)
					logger.Error("Could not get token.File from FileSet even with valid AST position", "filename", absFilename)
					// Add diagnostic for this internal error
					diag := Diagnostic{
						Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
						Severity: SeverityError,
						Source:   "deepcomplete-loader",
						Message:  fmt.Sprintf("Internal error: Could not find token file '%s' in fileset.", absFilename),
					}
					diagnostics = append(diagnostics, diag)
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
		// Add a diagnostic if the target file AST couldn't be found
		diag := Diagnostic{
			Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
			Severity: SeverityError,
			Source:   "deepcomplete-loader",
			Message:  fmt.Sprintf("Internal error: Could not locate AST for file '%s' in loaded packages.", absFilename),
		}
		diagnostics = append(diagnostics, diag)
		// Attempt to return any diagnostics found even if the target wasn't located
		return nil, nil, nil, diagnostics, loadErrors
	}

	// Final checks on the found package (if any)
	// Check if critical type information is missing, which indicates deeper issues
	if targetPkg.TypesInfo == nil {
		err := fmt.Errorf("type info (TypesInfo) is nil for target package %s", targetPkg.PkgPath)
		loadErrors = append(loadErrors, err)
		logger.Warn(err.Error())
		// Add diagnostic for missing type info
		diag := Diagnostic{
			Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
			Severity: SeverityWarning, // Warning, as some functionality might still work
			Source:   "deepcomplete-loader",
			Message:  fmt.Sprintf("Type checking information unavailable for package '%s'. Analysis may be incomplete.", targetPkg.PkgPath),
		}
		diagnostics = append(diagnostics, diag)
	}
	if targetPkg.Types == nil {
		err := fmt.Errorf("types (Types) is nil for target package %s", targetPkg.PkgPath)
		loadErrors = append(loadErrors, err)
		logger.Warn(err.Error())
		// Add diagnostic for missing types scope
		diag := Diagnostic{
			Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
			Severity: SeverityWarning,
			Source:   "deepcomplete-loader",
			Message:  fmt.Sprintf("Package scope information unavailable for package '%s'. Analysis may be incomplete.", targetPkg.PkgPath),
		}
		diagnostics = append(diagnostics, diag)
	}

	return targetPkg, targetFileAST, targetFile, diagnostics, loadErrors
}
