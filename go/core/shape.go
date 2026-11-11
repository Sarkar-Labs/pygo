package core

import "sync/atomic"

// GeneratedShapeCounter provides lock-free accounting for shape.
type GeneratedShapeCounter struct{ value atomic.Int64 }

func (c *GeneratedShapeCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedShapeCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedShapeCounter) Reset()            { c.value.Store(0) }
