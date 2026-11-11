package memory

import "sync/atomic"

// GeneratedProfilerCounter provides lock-free accounting for profiler.
type GeneratedProfilerCounter struct{ value atomic.Int64 }

func (c *GeneratedProfilerCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedProfilerCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedProfilerCounter) Reset()            { c.value.Store(0) }
