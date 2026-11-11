package core

import "sync/atomic"

// GeneratedScopeCounter provides lock-free accounting for scope.
type GeneratedScopeCounter struct{ value atomic.Int64 }

func (c *GeneratedScopeCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedScopeCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedScopeCounter) Reset()            { c.value.Store(0) }
