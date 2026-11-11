package device

import "sync/atomic"

// GeneratedNpuCounter provides lock-free accounting for npu.
type GeneratedNpuCounter struct{ value atomic.Int64 }

func (c *GeneratedNpuCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedNpuCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedNpuCounter) Reset()            { c.value.Store(0) }
