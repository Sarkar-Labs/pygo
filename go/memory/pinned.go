package memory

import "sync/atomic"

// GeneratedPinnedCounter provides lock-free accounting for pinned.
type GeneratedPinnedCounter struct{ value atomic.Int64 }

func (c *GeneratedPinnedCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPinnedCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPinnedCounter) Reset()            { c.value.Store(0) }
