package core

import "sync/atomic"

// GeneratedErrorCounter provides lock-free accounting for error.
type GeneratedErrorCounter struct{ value atomic.Int64 }

func (c *GeneratedErrorCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedErrorCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedErrorCounter) Reset()            { c.value.Store(0) }
