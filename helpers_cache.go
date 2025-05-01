// deepcomplete/helpers_cache.go
// Contains helper functions for memory caching (Ristretto).
// Cycle 3: Added explicit logger passing.
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

	// Check if analyzer or memory cache is available via the interface method
	if analyzer == nil || !analyzer.MemoryCacheEnabled() {
		logger.Debug("Memory cache check skipped (analyzer or cache disabled)", "key", cacheKey)
		result, err := computeFn()
		return result, false, err // Execute compute function directly
	}

	// --- Ristretto Cache Interaction ---
	// NOTE: Accessing the underlying Ristretto cache directly requires casting
	// the Analyzer interface back to the concrete type (*GoPackagesAnalyzer)
	// or adding Get/Set methods to the Analyzer interface itself.
	// For simplicity in this example, we assume direct access is possible
	// IF the concrete type is known or methods are added to the interface.

	// Example assuming we can get the concrete type (less ideal design):
	concreteAnalyzer, ok := analyzer.(*GoPackagesAnalyzer)
	if !ok || concreteAnalyzer.memoryCache == nil {
		logger.Warn("Memory cache enabled but cannot access concrete cache instance. Skipping cache.", "key", cacheKey)
		result, err := computeFn()
		return result, false, err
	}
	ristrettoCache := concreteAnalyzer.memoryCache // Access the Ristretto cache

	// 1. Check cache
	if cachedResult, found := ristrettoCache.Get(cacheKey); found {
		if typedResult, ok := cachedResult.(T); ok {
			logger.Debug("Ristretto cache hit", "key", cacheKey)
			return typedResult, true, nil // Return cached value
		}
		logger.Error("Ristretto cache type assertion failed", "key", cacheKey, "expected_type", fmt.Sprintf("%T", zero), "actual_type", fmt.Sprintf("%T", cachedResult))
		ristrettoCache.Del(cacheKey) // Delete invalid entry
	}
	logger.Debug("Ristretto cache miss", "key", cacheKey)

	// 2. Cache miss: Compute the value
	computedResult, err := computeFn()
	if err != nil {
		return zero, false, err // Do not cache errors
	}

	// 3. Store computed value in cache
	if cost <= 0 {
		cost = 1 // Ristretto cost must be positive
	}
	setOk := ristrettoCache.SetWithTTL(cacheKey, computedResult, cost, ttl)
	if !setOk {
		logger.Warn("Ristretto cache Set failed, item not cached", "key", cacheKey, "cost", cost)
	} else {
		logger.Debug("Ristretto cache set", "key", cacheKey, "cost", cost, "ttl", ttl)
	}

	return computedResult, false, nil // Return computed value
}
