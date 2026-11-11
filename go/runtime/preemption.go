package runtime

import "sync/atomic"

// GeneratedPreemptionCounter provides lock-free accounting for preemption.
type GeneratedPreemptionCounter struct{ value atomic.Int64 }

func (c *GeneratedPreemptionCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPreemptionCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPreemptionCounter) Reset()            { c.value.Store(0) }
