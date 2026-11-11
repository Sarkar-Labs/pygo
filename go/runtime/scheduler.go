package runtime

import "sync/atomic"

// GeneratedSchedulerCounter provides lock-free accounting for scheduler.
type GeneratedSchedulerCounter struct{ value atomic.Int64 }

func (c *GeneratedSchedulerCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedSchedulerCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedSchedulerCounter) Reset()            { c.value.Store(0) }
