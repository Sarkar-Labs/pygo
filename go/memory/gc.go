package memory

import "sync/atomic"

// GeneratedGcCounter provides lock-free accounting for gc.
type GeneratedGcCounter struct{ value atomic.Int64 }

func (c *GeneratedGcCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedGcCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedGcCounter) Reset()            { c.value.Store(0) }
