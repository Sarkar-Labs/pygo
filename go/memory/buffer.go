package memory

import "sync/atomic"

// GeneratedBufferCounter provides lock-free accounting for buffer.
type GeneratedBufferCounter struct{ value atomic.Int64 }

func (c *GeneratedBufferCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedBufferCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedBufferCounter) Reset()            { c.value.Store(0) }
