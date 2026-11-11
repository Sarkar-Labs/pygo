package runtime

import "sync/atomic"

// GeneratedMemoryOrderCounter provides lock-free accounting for memory order.
type GeneratedMemoryOrderCounter struct{ value atomic.Int64 }

func (c *GeneratedMemoryOrderCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedMemoryOrderCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedMemoryOrderCounter) Reset()            { c.value.Store(0) }
