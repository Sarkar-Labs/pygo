package memory

import "sync/atomic"

// GeneratedPhysicalCounter provides lock-free accounting for physical.
type GeneratedPhysicalCounter struct{ value atomic.Int64 }

func (c *GeneratedPhysicalCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPhysicalCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPhysicalCounter) Reset()            { c.value.Store(0) }
