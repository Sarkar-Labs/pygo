package device

import "sync/atomic"

// GeneratedDeviceManagerCounter provides lock-free accounting for device manager.
type GeneratedDeviceManagerCounter struct{ value atomic.Int64 }

func (c *GeneratedDeviceManagerCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDeviceManagerCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDeviceManagerCounter) Reset()            { c.value.Store(0) }
