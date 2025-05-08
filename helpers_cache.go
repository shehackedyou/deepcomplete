// deepcomplete/helpers_cache.go
// Contains helper functions for memory caching (Ristretto).
package deepcomplete

import (
	"fmt"
	"log/slog"
	"time"
	// Ristretto import removed as it's now abstracted by the Analyzer interface
)

// ============================================================================
// Memory Cache Helpers
// ============================================================================

// generateCacheKey creates a key for the memory cache based on context.
// Key includes a prefix, file path, version, and cursor position for fine-grained caching.
func generateCacheKey(prefix string, info *AstContextInfo) string {
	filePath := info.FilePath
	cursorPos := info.CursorPos
	if filePath == "" {
		filePath = "[unknown-path]"
	}
	if !cursorPos.IsValid() {
		cursorPos = -1 // Use -1 for invalid position
	}
	// Format: prefix:filepath:version:cursorPos
	return fmt.Sprintf("%s:%s:%d:%d", prefix, filePath, info.Version, cursorPos)
}

// withMemoryCache wraps a function call with caching logic using the Analyzer interface.
// Tries to fetch from cache using cacheKey. If miss, calls computeFn,
// stores the result with cost and ttl, and returns it.
// Returns the result (cached or computed), a boolean indicating cache hit, and any error from computeFn.
func withMemoryCache[T any](
	analyzer Analyzer, // Use the Analyzer interface
	cacheKey string,
	cost int64, // Estimated cost for Ristretto (e.g., size in bytes, or just 1)
	ttl time.Duration, // Time-to-live for the cache entry
	computeFn func() (T, error), // Function to compute the value if cache miss
	logger *slog.Logger, // Accept logger explicitly
) (T, bool, error) {
	var zero T // Zero value for the return type T
	if logger == nil {
		logger = slog.Default() // Fallback if logger is nil
	}
	cacheLogger := logger.With("cache_key", cacheKey)

	// Check if analyzer or memory cache is available via the interface method
	if analyzer == nil || !analyzer.MemoryCacheEnabled() {
		cacheLogger.Debug("Memory cache check skipped (analyzer or cache disabled)")
		result, err := computeFn()
		return result, false, err // Execute compute function directly
	}

	// --- Cache Interaction via Analyzer Interface ---

	// 1. Check cache using the interface method
	cachedResult, found := analyzer.GetMemoryCache(cacheKey) // Use interface method
	if found {
		// Attempt type assertion on the cached result
		if typedResult, ok := cachedResult.(T); ok {
			cacheLogger.Debug("Memory cache hit")
			return typedResult, true, nil // Return cached value
		}
		// Type assertion failed - indicates corrupted or mismatched cache entry
		cacheLogger.Error("Memory cache type assertion failed", "expected_type", fmt.Sprintf("%T", zero), "actual_type", fmt.Sprintf("%T", cachedResult))
		// TODO: Consider adding an Invalidate(key) method to the interface?
		// For now, we just treat it as a miss. The invalid entry will expire eventually.
	} else {
		cacheLogger.Debug("Memory cache miss")
	}

	// 2. Cache miss: Compute the value
	computedResult, err := computeFn()
	if err != nil {
		// Do not cache errors, return immediately
		return zero, false, err
	}

	// 3. Store computed value in cache using the interface method
	if cost <= 0 {
		cost = 1 // Ristretto cost must be positive
	}
	setOk := analyzer.SetMemoryCache(cacheKey, computedResult, cost, ttl) // Use interface method
	if !setOk {
		// Set can fail if the item is too large or other constraints aren't met
		cacheLogger.Warn("Memory cache Set failed, item not cached", "cost", cost, "ttl", ttl)
	} else {
		cacheLogger.Debug("Memory cache set successful", "cost", cost, "ttl", ttl)
	}

	// Return the newly computed value
	return computedResult, false, nil
}

// Helper to estimate cost for simple types (can be expanded)
func estimateCost(v any) int64 {
	switch val := v.(type) {
	case string:
		return int64(len(val))
	case []byte:
		return int64(len(val))
	case []string:
		cost := int64(0)
		for _, s := range val {
			cost += int64(len(s))
		}
		return cost
	// Add cases for other types if needed (e.g., slices of structs)
	default:
		return 1 // Default cost if size is unknown or hard to calculate
	}
}
