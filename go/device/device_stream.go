package device

import "sync/atomic"

// GeneratedDeviceStreamCounter provides lock-free accounting for device stream.
type GeneratedDeviceStreamCounter struct{ value atomic.Int64 }

func (c *GeneratedDeviceStreamCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDeviceStreamCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDeviceStreamCounter) Reset()            { c.value.Store(0) }
