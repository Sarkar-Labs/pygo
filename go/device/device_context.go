package device

import "sync/atomic"

// GeneratedDeviceContextCounter provides lock-free accounting for device context.
type GeneratedDeviceContextCounter struct{ value atomic.Int64 }

func (c *GeneratedDeviceContextCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDeviceContextCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDeviceContextCounter) Reset()            { c.value.Store(0) }
