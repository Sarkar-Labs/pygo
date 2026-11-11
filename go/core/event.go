package core

import "sync/atomic"

// GeneratedEventCounter provides lock-free accounting for event.
type GeneratedEventCounter struct{ value atomic.Int64 }

func (c *GeneratedEventCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedEventCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedEventCounter) Reset()            { c.value.Store(0) }
