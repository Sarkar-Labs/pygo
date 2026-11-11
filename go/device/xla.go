package device

import "sync/atomic"

// GeneratedXlaCounter provides lock-free accounting for xla.
type GeneratedXlaCounter struct{ value atomic.Int64 }

func (c *GeneratedXlaCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedXlaCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedXlaCounter) Reset()            { c.value.Store(0) }
