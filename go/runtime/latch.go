package runtime

import "sync/atomic"

// GeneratedLatchCounter provides lock-free accounting for latch.
type GeneratedLatchCounter struct{ value atomic.Int64 }

func (c *GeneratedLatchCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedLatchCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedLatchCounter) Reset()            { c.value.Store(0) }
