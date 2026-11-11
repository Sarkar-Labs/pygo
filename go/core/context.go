package core

import "sync/atomic"

// GeneratedContextCounter provides lock-free accounting for context.
type GeneratedContextCounter struct{ value atomic.Int64 }

func (c *GeneratedContextCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedContextCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedContextCounter) Reset()            { c.value.Store(0) }
