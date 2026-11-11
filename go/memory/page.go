package memory

import "sync/atomic"

// GeneratedPageCounter provides lock-free accounting for page.
type GeneratedPageCounter struct{ value atomic.Int64 }

func (c *GeneratedPageCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPageCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPageCounter) Reset()            { c.value.Store(0) }
