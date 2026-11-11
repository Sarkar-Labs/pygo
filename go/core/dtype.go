package core

import "sync/atomic"

// GeneratedDtypeCounter provides lock-free accounting for dtype.
type GeneratedDtypeCounter struct{ value atomic.Int64 }

func (c *GeneratedDtypeCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDtypeCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDtypeCounter) Reset()            { c.value.Store(0) }
