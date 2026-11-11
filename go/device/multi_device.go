package device

import "sync/atomic"

// GeneratedMultiDeviceCounter provides lock-free accounting for multi device.
type GeneratedMultiDeviceCounter struct{ value atomic.Int64 }

func (c *GeneratedMultiDeviceCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedMultiDeviceCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedMultiDeviceCounter) Reset()            { c.value.Store(0) }
