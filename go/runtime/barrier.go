package runtime

import "sync/atomic"

// GeneratedBarrierCounter provides lock-free accounting for barrier.
type GeneratedBarrierCounter struct{ value atomic.Int64 }

func (c *GeneratedBarrierCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedBarrierCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedBarrierCounter) Reset()            { c.value.Store(0) }
