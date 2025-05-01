// deepcomplete/errors.go
// Contains exported error definitions for the deepcomplete package.
package deepcomplete

import "errors"

// =============================================================================
// Exported Errors
// =============================================================================

// Note: Error variables moved here from deepcomplete.go in Cycle 1.

var (
	// ErrAnalysisFailed indicates non-fatal errors occurred during code analysis.
	// Further details should be available in AstContextInfo.AnalysisErrors.
	ErrAnalysisFailed = errors.New("code analysis failed")

	// ErrOllamaUnavailable indicates failure communicating with the Ollama API.
	ErrOllamaUnavailable = errors.New("ollama API unavailable")

	// ErrStreamProcessing indicates an error reading or processing the LLM response stream.
	ErrStreamProcessing = errors.New("error processing LLM stream")

	// ErrConfig indicates non-fatal errors during config loading or processing.
	ErrConfig = errors.New("configuration error")

	// ErrInvalidConfig indicates a configuration value is invalid after validation.
	ErrInvalidConfig = errors.New("invalid configuration")

	// ErrCache indicates a general cache operation failure.
	ErrCache = errors.New("cache operation failed")

	// ErrCacheRead indicates failure reading from the cache.
	ErrCacheRead = errors.New("cache read failed")

	// ErrCacheWrite indicates failure writing to the cache.
	ErrCacheWrite = errors.New("cache write failed")

	// ErrCacheDecode indicates failure decoding data read from the cache.
	ErrCacheDecode = errors.New("cache decode failed")

	// ErrCacheEncode indicates failure encoding data for writing to the cache.
	ErrCacheEncode = errors.New("cache encode failed")

	// ErrCacheHash indicates failure calculating file hashes for cache validation.
	ErrCacheHash = errors.New("cache hash calculation failed")

	// ErrPositionConversion indicates failure converting between position formats (e.g., LSP <-> byte offset).
	ErrPositionConversion = errors.New("position conversion failed")

	// ErrInvalidPositionInput indicates input position values (line/col) are invalid.
	ErrInvalidPositionInput = errors.New("invalid input position")

	// ErrPositionOutOfRange indicates a position is outside the valid bounds of the file or line.
	ErrPositionOutOfRange = errors.New("position out of range")

	// ErrInvalidUTF8 indicates an invalid UTF-8 sequence was encountered during processing.
	ErrInvalidUTF8 = errors.New("invalid utf-8 sequence")

	// ErrInvalidURI indicates a document URI is invalid or uses an unsupported scheme.
	ErrInvalidURI = errors.New("invalid document URI")
)
