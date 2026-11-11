package memory

import "sync/atomic"

// GeneratedVirtualCounter provides lock-free accounting for virtual.
type GeneratedVirtualCounter struct{ value atomic.Int64 }

func (c *GeneratedVirtualCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedVirtualCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedVirtualCounter) Reset()            { c.value.Store(0) }
