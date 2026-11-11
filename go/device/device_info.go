package device

import "sync/atomic"

// GeneratedDeviceInfoCounter provides lock-free accounting for device info.
type GeneratedDeviceInfoCounter struct{ value atomic.Int64 }

func (c *GeneratedDeviceInfoCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDeviceInfoCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDeviceInfoCounter) Reset()            { c.value.Store(0) }
