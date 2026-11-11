package memory

import "sync/atomic"

// GeneratedUnifiedCounter provides lock-free accounting for unified.
type GeneratedUnifiedCounter struct{ value atomic.Int64 }

func (c *GeneratedUnifiedCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedUnifiedCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedUnifiedCounter) Reset()            { c.value.Store(0) }
