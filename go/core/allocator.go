package core

import "sync/atomic"

// GeneratedAllocatorCounter provides lock-free accounting for allocator.
type GeneratedAllocatorCounter struct{ value atomic.Int64 }

func (c *GeneratedAllocatorCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedAllocatorCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedAllocatorCounter) Reset()            { c.value.Store(0) }
