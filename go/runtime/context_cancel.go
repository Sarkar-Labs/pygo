package runtime

import "sync/atomic"

// GeneratedContextCancelCounter provides lock-free accounting for context cancel.
type GeneratedContextCancelCounter struct{ value atomic.Int64 }

func (c *GeneratedContextCancelCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedContextCancelCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedContextCancelCounter) Reset()            { c.value.Store(0) }
