package runtime

import "sync/atomic"

// GeneratedParallelForCounter provides lock-free accounting for parallel for.
type GeneratedParallelForCounter struct{ value atomic.Int64 }

func (c *GeneratedParallelForCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedParallelForCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedParallelForCounter) Reset()            { c.value.Store(0) }
