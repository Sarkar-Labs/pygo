package device

import "sync/atomic"

// GeneratedFpgaCounter provides lock-free accounting for fpga.
type GeneratedFpgaCounter struct{ value atomic.Int64 }

func (c *GeneratedFpgaCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedFpgaCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedFpgaCounter) Reset()            { c.value.Store(0) }
