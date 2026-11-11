package models

import "time"

// GeneratedLlamaXlarge records execution state for the llama xlarge component.
type GeneratedLlamaXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaXlarge() *GeneratedLlamaXlarge {
	return &GeneratedLlamaXlarge{Name: "llama_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
