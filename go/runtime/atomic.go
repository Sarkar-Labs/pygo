package runtime

import "sync/atomic"

// GeneratedAtomicCounter provides lock-free accounting for atomic.
type GeneratedAtomicCounter struct{ value atomic.Int64 }

func (c *GeneratedAtomicCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedAtomicCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedAtomicCounter) Reset()            { c.value.Store(0) }
