package memory

import "sync/atomic"

// GeneratedMemoryStatsCounter provides lock-free accounting for memory stats.
type GeneratedMemoryStatsCounter struct{ value atomic.Int64 }

func (c *GeneratedMemoryStatsCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedMemoryStatsCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedMemoryStatsCounter) Reset()            { c.value.Store(0) }
