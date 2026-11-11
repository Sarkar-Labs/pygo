package device

import "sync/atomic"

// GeneratedDevicePropertiesCounter provides lock-free accounting for device properties.
type GeneratedDevicePropertiesCounter struct{ value atomic.Int64 }

func (c *GeneratedDevicePropertiesCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDevicePropertiesCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDevicePropertiesCounter) Reset()            { c.value.Store(0) }
