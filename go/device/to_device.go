package device

import "sync/atomic"

// GeneratedToDeviceCounter provides lock-free accounting for to device.
type GeneratedToDeviceCounter struct{ value atomic.Int64 }

func (c *GeneratedToDeviceCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedToDeviceCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedToDeviceCounter) Reset()            { c.value.Store(0) }
