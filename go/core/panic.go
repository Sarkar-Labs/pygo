package core

import "sync/atomic"

// GeneratedPanicCounter provides lock-free accounting for panic.
type GeneratedPanicCounter struct{ value atomic.Int64 }

func (c *GeneratedPanicCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPanicCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPanicCounter) Reset()            { c.value.Store(0) }
