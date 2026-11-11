package device

import "sync/atomic"

// GeneratedGpuCounter provides lock-free accounting for gpu.
type GeneratedGpuCounter struct{ value atomic.Int64 }

func (c *GeneratedGpuCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedGpuCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedGpuCounter) Reset()            { c.value.Store(0) }
