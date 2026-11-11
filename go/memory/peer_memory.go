package memory

import "sync/atomic"

// GeneratedPeerMemoryCounter provides lock-free accounting for peer memory.
type GeneratedPeerMemoryCounter struct{ value atomic.Int64 }

func (c *GeneratedPeerMemoryCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPeerMemoryCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPeerMemoryCounter) Reset()            { c.value.Store(0) }
