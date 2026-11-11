package memory

import "sync/atomic"

// GeneratedHostMemoryCounter provides lock-free accounting for host memory.
type GeneratedHostMemoryCounter struct{ value atomic.Int64 }

func (c *GeneratedHostMemoryCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedHostMemoryCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedHostMemoryCounter) Reset()            { c.value.Store(0) }
