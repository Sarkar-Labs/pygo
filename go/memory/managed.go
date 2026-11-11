package memory

import "sync/atomic"

// GeneratedManagedCounter provides lock-free accounting for managed.
type GeneratedManagedCounter struct{ value atomic.Int64 }

func (c *GeneratedManagedCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedManagedCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedManagedCounter) Reset()            { c.value.Store(0) }
