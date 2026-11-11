package memory

import "sync/atomic"

// GeneratedArenaCounter provides lock-free accounting for arena.
type GeneratedArenaCounter struct{ value atomic.Int64 }

func (c *GeneratedArenaCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedArenaCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedArenaCounter) Reset()            { c.value.Store(0) }
