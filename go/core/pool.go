package core

import "sync/atomic"

// GeneratedPoolCounter provides lock-free accounting for pool.
type GeneratedPoolCounter struct{ value atomic.Int64 }

func (c *GeneratedPoolCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPoolCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPoolCounter) Reset()            { c.value.Store(0) }
