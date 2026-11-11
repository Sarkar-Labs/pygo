package device

import "sync/atomic"

// GeneratedTpuCounter provides lock-free accounting for tpu.
type GeneratedTpuCounter struct{ value atomic.Int64 }

func (c *GeneratedTpuCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedTpuCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedTpuCounter) Reset()            { c.value.Store(0) }
