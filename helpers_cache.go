// deepcomplete/helpers_cache.go
package deepcomplete

import (
	"fmt"
	"log/slog"
	"time"
)

// ============================================================================
// Memory Cache Helpers (Conceptual - Cycle 9)
// ============================================================================

// generateCacheKey creates a key for the memory cache based on context.
// Used conceptually for caching analysis steps.
func generateCacheKey(prefix string, info *AstContextInfo) string {
	// Key includes file path, version, and cursor position for fine-grained caching
	return fmt.Sprintf("%s:%s:%d:%d", prefix, info.FilePath, info.Version, info.CursorPos)
}

// withMemoryCache wraps a function call with Ristretto caching logic.
// Tries to fetch from cache using cacheKey. If miss, calls computeFn,
// stores the result with cost and ttl, and returns it.
// Returns the result (cached or computed), a boolean indicating cache hit, and any error from computeFn.
func withMemoryCache[T any](
	analyzer *GoPackagesAnalyzer,
	cacheKey string,
	cost int64, // Estimated cost for Ristretto (e.g., size in bytes, or just 1)
	ttl time.Duration, // Time-to-live for the cache entry
	computeFn func() (T, error), // Function to compute the value if cache miss
	logger *slog.Logger,
) (T, bool, error) {
	var zero T // Zero value for the return type T

	// Check if analyzer or memory cache is available
	if analyzer == nil || !analyzer.MemoryCacheEnabled() {
		logger.Debug("Memory cache check skipped (analyzer or cache disabled)", "key", cacheKey)
		result, err := computeFn()
		return result, false, err // Execute compute function directly
	}

	// 1. Check cache
	if cachedResult, found := analyzer.memoryCache.Get(cacheKey); found {
		// Attempt to cast cached result to the expected type T
		if typedResult, ok := cachedResult.(T); ok {
			logger.Debug("Ristretto cache hit", "key", cacheKey)
			return typedResult, true, nil // Return cached value
		}
		// Cache contained wrong type - this indicates a programming error (e.g., inconsistent key usage)
		logger.Error("Ristretto cache type assertion failed", "key", cacheKey, "expected_type", fmt.Sprintf("%T", zero), "actual_type", fmt.Sprintf("%T", cachedResult))
		analyzer.memoryCache.Del(cacheKey) // Delete the invalid entry
		// Proceed to compute as if it was a miss
	}
	logger.Debug("Ristretto cache miss", "key", cacheKey)

	// 2. Cache miss: Compute the value
	computedResult, err := computeFn()
	if err != nil {
		// Do not cache errors
		return zero, false, err // Return zero value and the error
	}

	// 3. Store computed value in cache (only if no error occurred)
	if cost <= 0 {
		cost = 1 // Ristretto cost must be positive
	}
	setOk := analyzer.memoryCache.SetWithTTL(cacheKey, computedResult, cost, ttl)
	if !setOk {
		// Set might fail if item cost > max cache cost, or other internal reasons
		logger.Warn("Ristretto cache Set failed, item not cached", "key", cacheKey, "cost", cost)
	} else {
		logger.Debug("Ristretto cache set", "key", cacheKey, "cost", cost, "ttl", ttl)
	}

	// Return the newly computed value
	return computedResult, false, nil
}
