package device

import "sync/atomic"

// GeneratedMpsCounter provides lock-free accounting for mps.
type GeneratedMpsCounter struct{ value atomic.Int64 }

func (c *GeneratedMpsCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedMpsCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedMpsCounter) Reset()            { c.value.Store(0) }
