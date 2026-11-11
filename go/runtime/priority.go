package runtime

import "sync/atomic"

// GeneratedPriorityCounter provides lock-free accounting for priority.
type GeneratedPriorityCounter struct{ value atomic.Int64 }

func (c *GeneratedPriorityCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPriorityCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPriorityCounter) Reset()            { c.value.Store(0) }
