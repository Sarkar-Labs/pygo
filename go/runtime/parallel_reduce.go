package runtime

import "sync/atomic"

// GeneratedParallelReduceCounter provides lock-free accounting for parallel reduce.
type GeneratedParallelReduceCounter struct{ value atomic.Int64 }

func (c *GeneratedParallelReduceCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedParallelReduceCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedParallelReduceCounter) Reset()            { c.value.Store(0) }
