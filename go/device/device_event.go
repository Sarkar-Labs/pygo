package device

import "sync/atomic"

// GeneratedDeviceEventCounter provides lock-free accounting for device event.
type GeneratedDeviceEventCounter struct{ value atomic.Int64 }

func (c *GeneratedDeviceEventCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDeviceEventCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDeviceEventCounter) Reset()            { c.value.Store(0) }
