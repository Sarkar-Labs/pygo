package memory

import "sync/atomic"

// GeneratedSharedCounter provides lock-free accounting for shared.
type GeneratedSharedCounter struct{ value atomic.Int64 }

func (c *GeneratedSharedCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedSharedCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedSharedCounter) Reset()            { c.value.Store(0) }
