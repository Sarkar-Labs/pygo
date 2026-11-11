package runtime

import "sync/atomic"

// GeneratedThreadPoolCounter provides lock-free accounting for thread pool.
type GeneratedThreadPoolCounter struct{ value atomic.Int64 }

func (c *GeneratedThreadPoolCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedThreadPoolCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedThreadPoolCounter) Reset()            { c.value.Store(0) }
