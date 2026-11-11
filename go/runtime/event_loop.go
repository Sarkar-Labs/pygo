package runtime

import "sync/atomic"

// GeneratedEventLoopCounter provides lock-free accounting for event loop.
type GeneratedEventLoopCounter struct{ value atomic.Int64 }

func (c *GeneratedEventLoopCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedEventLoopCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedEventLoopCounter) Reset()            { c.value.Store(0) }
