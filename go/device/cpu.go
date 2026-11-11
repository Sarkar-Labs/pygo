package device

import "sync/atomic"

// GeneratedCpuCounter provides lock-free accounting for cpu.
type GeneratedCpuCounter struct{ value atomic.Int64 }

func (c *GeneratedCpuCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedCpuCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedCpuCounter) Reset()            { c.value.Store(0) }
