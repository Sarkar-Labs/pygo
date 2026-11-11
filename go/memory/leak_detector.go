package memory

import "sync/atomic"

// GeneratedLeakDetectorCounter provides lock-free accounting for leak detector.
type GeneratedLeakDetectorCounter struct{ value atomic.Int64 }

func (c *GeneratedLeakDetectorCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedLeakDetectorCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedLeakDetectorCounter) Reset()            { c.value.Store(0) }
