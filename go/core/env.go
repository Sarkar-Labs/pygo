package core

import "sync/atomic"

// GeneratedEnvCounter provides lock-free accounting for env.
type GeneratedEnvCounter struct{ value atomic.Int64 }

func (c *GeneratedEnvCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedEnvCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedEnvCounter) Reset()            { c.value.Store(0) }
