// deepcomplete/helpers_loader.go
// Contains helper functions specifically for loading packages using go/packages.
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

	loadCfg := &packages.Config{
		Context: ctx,
		Dir:     dir,
		Fset:    fset,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedTypes | packages.NeedTypesSizes |
			packages.NeedSyntax | packages.NeedTypesInfo,
		Tests: false,
		// Pass logger to packages.Load for its internal logging
		Logf: func(format string, args ...interface{}) { logger.Debug(fmt.Sprintf(format, args...)) },
	}

	logger.Debug("Calling packages.Load")
	pkgs, err := packages.Load(loadCfg, fmt.Sprintf("file=%s", absFilename))
	if err != nil {
		loadErr := fmt.Errorf("packages.Load failed critically: %w", err)
		loadErrors = append(loadErrors, loadErr)
		logger.Error("packages.Load failed critically", "error", err)
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			diag := Diagnostic{
				Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
				Severity: SeverityError,
				Source:   "deepcomplete-loader",
				Message:  fmt.Sprintf("Critical error loading package: %v", err),
			}
			diagnostics = append(diagnostics, diag)
		}
		return nil, nil, nil, diagnostics, loadErrors
	}

	if len(pkgs) == 0 {
		loadErr := errors.New("packages.Load returned no packages")
		loadErrors = append(loadErrors, loadErr)
		logger.Warn(loadErr.Error())
		diag := Diagnostic{
			Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
			Severity: SeverityWarning,
			Source:   "deepcomplete-loader",
			Message:  fmt.Sprintf("Could not load package information for file: %s", absFilename),
		}
		diagnostics = append(diagnostics, diag)
		return nil, nil, nil, diagnostics, loadErrors
	}

	var targetPkg *packages.Package
	var targetFileAST *ast.File
	var targetFile *token.File

	for _, p := range pkgs {
		if p == nil {
			continue
		}
		for i := range p.Errors {
			pkgErr := p.Errors[i]
			loadErrors = append(loadErrors, &pkgErr)
			logger.Warn("Package loading error encountered", "package", p.PkgPath, "error", pkgErr.Error())
			// Pass logger to packagesErrorToDiagnostic
			diag := packagesErrorToDiagnostic(pkgErr, fset, logger) // Pass logger here
			if diag != nil {
				diagnostics = append(diagnostics, *diag)
			}
		}
	}

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
			filePos := fset.Position(astFile.Pos())
			if !filePos.IsValid() {
				logger.Warn("AST file has invalid position", "package", p.PkgPath, "index", i)
				continue
			}
			astFilePath, _ := filepath.Abs(filePos.Filename)
			logger.Debug("Comparing file paths", "ast_file_path", astFilePath, "target_file_path", absFilename)
			if astFilePath == absFilename {
				targetPkg = p
				targetFileAST = astFile
				targetFile = fset.File(astFile.Pos())
				if targetFile == nil {
					fileNotFoundErr := fmt.Errorf("failed to get token.File for AST file %s from FileSet", absFilename)
					loadErrors = append(loadErrors, fileNotFoundErr)
					logger.Error("Could not get token.File from FileSet even with valid AST position", "filename", absFilename)
					diag := Diagnostic{
						Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
						Severity: SeverityError,
						Source:   "deepcomplete-loader",
						Message:  fmt.Sprintf("Internal error: Could not find token file '%s' in fileset.", absFilename),
					}
					diagnostics = append(diagnostics, diag)
				}
				logger.Debug("Found target file in package", "package", p.PkgPath)
				goto foundTarget
			}
		}
	}

foundTarget:
	if targetPkg == nil {
		err := fmt.Errorf("target file %s not found in loaded packages syntax trees", absFilename)
		loadErrors = append(loadErrors, err)
		logger.Warn(err.Error())
		diag := Diagnostic{
			Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
			Severity: SeverityError,
			Source:   "deepcomplete-loader",
			Message:  fmt.Sprintf("Internal error: Could not locate AST for file '%s' in loaded packages.", absFilename),
		}
		diagnostics = append(diagnostics, diag)
		return nil, nil, nil, diagnostics, loadErrors
	}

	if targetPkg.TypesInfo == nil {
		err := fmt.Errorf("type info (TypesInfo) is nil for target package %s", targetPkg.PkgPath)
		loadErrors = append(loadErrors, err)
		logger.Warn(err.Error())
		diag := Diagnostic{
			Range:    Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}},
			Severity: SeverityWarning,
			Source:   "deepcomplete-loader",
			Message:  fmt.Sprintf("Type checking information unavailable for package '%s'. Analysis may be incomplete.", targetPkg.PkgPath),
		}
		diagnostics = append(diagnostics, diag)
	}
	if targetPkg.Types == nil {
		err := fmt.Errorf("types (Types) is nil for target package %s", targetPkg.PkgPath)
		loadErrors = append(loadErrors, err)
		logger.Warn(err.Error())
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
