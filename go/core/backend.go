package core

import "sync/atomic"

// GeneratedBackendCounter provides lock-free accounting for backend.
type GeneratedBackendCounter struct{ value atomic.Int64 }

func (c *GeneratedBackendCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedBackendCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedBackendCounter) Reset()            { c.value.Store(0) }
