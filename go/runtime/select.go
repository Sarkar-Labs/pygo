package runtime

import "sync/atomic"

// GeneratedSelectCounter provides lock-free accounting for select.
type GeneratedSelectCounter struct{ value atomic.Int64 }

func (c *GeneratedSelectCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedSelectCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedSelectCounter) Reset()            { c.value.Store(0) }
