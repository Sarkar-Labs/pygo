package runtime

import "sync/atomic"

// GeneratedSemaphoreCounter provides lock-free accounting for semaphore.
type GeneratedSemaphoreCounter struct{ value atomic.Int64 }

func (c *GeneratedSemaphoreCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedSemaphoreCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedSemaphoreCounter) Reset()            { c.value.Store(0) }
