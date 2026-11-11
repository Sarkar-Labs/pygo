package runtime

import "sync/atomic"

// GeneratedRwlockCounter provides lock-free accounting for rwlock.
type GeneratedRwlockCounter struct{ value atomic.Int64 }

func (c *GeneratedRwlockCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedRwlockCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedRwlockCounter) Reset()            { c.value.Store(0) }
