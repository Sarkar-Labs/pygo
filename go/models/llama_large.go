package models

import "time"

// GeneratedLlamaLarge records execution state for the llama large component.
type GeneratedLlamaLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaLarge() *GeneratedLlamaLarge {
	return &GeneratedLlamaLarge{Name: "llama_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
