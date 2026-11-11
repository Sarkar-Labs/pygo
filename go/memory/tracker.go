package memory

import "sync/atomic"

// GeneratedTrackerCounter provides lock-free accounting for tracker.
type GeneratedTrackerCounter struct{ value atomic.Int64 }

func (c *GeneratedTrackerCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedTrackerCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedTrackerCounter) Reset()            { c.value.Store(0) }
