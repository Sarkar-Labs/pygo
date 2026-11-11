package memory

import "sync/atomic"

// GeneratedDeviceMemoryCounter provides lock-free accounting for device memory.
type GeneratedDeviceMemoryCounter struct{ value atomic.Int64 }

func (c *GeneratedDeviceMemoryCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDeviceMemoryCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDeviceMemoryCounter) Reset()            { c.value.Store(0) }
