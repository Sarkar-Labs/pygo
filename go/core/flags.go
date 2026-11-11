package core

import "sync/atomic"

// GeneratedFlagsCounter provides lock-free accounting for flags.
type GeneratedFlagsCounter struct{ value atomic.Int64 }

func (c *GeneratedFlagsCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedFlagsCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedFlagsCounter) Reset()            { c.value.Store(0) }
