package memory

import "sync/atomic"

// GeneratedDefragCounter provides lock-free accounting for defrag.
type GeneratedDefragCounter struct{ value atomic.Int64 }

func (c *GeneratedDefragCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDefragCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDefragCounter) Reset()            { c.value.Store(0) }
