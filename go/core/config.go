package core

import "sync/atomic"

// GeneratedConfigCounter provides lock-free accounting for config.
type GeneratedConfigCounter struct{ value atomic.Int64 }

func (c *GeneratedConfigCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedConfigCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedConfigCounter) Reset()            { c.value.Store(0) }
