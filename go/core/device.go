package core

import "sync/atomic"

// GeneratedDeviceCounter provides lock-free accounting for device.
type GeneratedDeviceCounter struct{ value atomic.Int64 }

func (c *GeneratedDeviceCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDeviceCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDeviceCounter) Reset()            { c.value.Store(0) }
