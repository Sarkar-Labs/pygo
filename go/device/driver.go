package device

import "sync/atomic"

// GeneratedDriverCounter provides lock-free accounting for driver.
type GeneratedDriverCounter struct{ value atomic.Int64 }

func (c *GeneratedDriverCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedDriverCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedDriverCounter) Reset()            { c.value.Store(0) }
