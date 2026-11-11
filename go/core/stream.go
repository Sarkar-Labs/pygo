package core

import "sync/atomic"

// GeneratedStreamCounter provides lock-free accounting for stream.
type GeneratedStreamCounter struct{ value atomic.Int64 }

func (c *GeneratedStreamCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedStreamCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedStreamCounter) Reset()            { c.value.Store(0) }
