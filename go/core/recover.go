package core

import "sync/atomic"

// GeneratedRecoverCounter provides lock-free accounting for recover.
type GeneratedRecoverCounter struct{ value atomic.Int64 }

func (c *GeneratedRecoverCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedRecoverCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedRecoverCounter) Reset()            { c.value.Store(0) }
