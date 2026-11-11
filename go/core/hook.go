package core

import "sync/atomic"

// GeneratedHookCounter provides lock-free accounting for hook.
type GeneratedHookCounter struct{ value atomic.Int64 }

func (c *GeneratedHookCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedHookCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedHookCounter) Reset()            { c.value.Store(0) }
