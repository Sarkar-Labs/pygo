package core

import "sync/atomic"

// GeneratedRegistryCounter provides lock-free accounting for registry.
type GeneratedRegistryCounter struct{ value atomic.Int64 }

func (c *GeneratedRegistryCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedRegistryCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedRegistryCounter) Reset()            { c.value.Store(0) }
