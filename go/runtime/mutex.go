package runtime

import "sync/atomic"

// GeneratedMutexCounter provides lock-free accounting for mutex.
type GeneratedMutexCounter struct{ value atomic.Int64 }

func (c *GeneratedMutexCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedMutexCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedMutexCounter) Reset()            { c.value.Store(0) }
