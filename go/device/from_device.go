package device

import "sync/atomic"

// GeneratedFromDeviceCounter provides lock-free accounting for from device.
type GeneratedFromDeviceCounter struct{ value atomic.Int64 }

func (c *GeneratedFromDeviceCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedFromDeviceCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedFromDeviceCounter) Reset()            { c.value.Store(0) }
