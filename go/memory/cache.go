package memory

import "sync/atomic"

// GeneratedCacheCounter provides lock-free accounting for cache.
type GeneratedCacheCounter struct{ value atomic.Int64 }

func (c *GeneratedCacheCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedCacheCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedCacheCounter) Reset()            { c.value.Store(0) }
