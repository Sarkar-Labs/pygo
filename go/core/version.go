package core

import "sync/atomic"

// GeneratedVersionCounter provides lock-free accounting for version.
type GeneratedVersionCounter struct{ value atomic.Int64 }

func (c *GeneratedVersionCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedVersionCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedVersionCounter) Reset()            { c.value.Store(0) }
