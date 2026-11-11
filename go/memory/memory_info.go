package memory

import "sync/atomic"

// GeneratedMemoryInfoCounter provides lock-free accounting for memory info.
type GeneratedMemoryInfoCounter struct{ value atomic.Int64 }

func (c *GeneratedMemoryInfoCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedMemoryInfoCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedMemoryInfoCounter) Reset()            { c.value.Store(0) }
