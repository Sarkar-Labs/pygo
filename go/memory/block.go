package memory

import "sync/atomic"

// GeneratedBlockCounter provides lock-free accounting for block.
type GeneratedBlockCounter struct{ value atomic.Int64 }

func (c *GeneratedBlockCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedBlockCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedBlockCounter) Reset()            { c.value.Store(0) }
