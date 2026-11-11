package runtime

import "sync/atomic"

// GeneratedTaskQueueCounter provides lock-free accounting for task queue.
type GeneratedTaskQueueCounter struct{ value atomic.Int64 }

func (c *GeneratedTaskQueueCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedTaskQueueCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedTaskQueueCounter) Reset()            { c.value.Store(0) }
