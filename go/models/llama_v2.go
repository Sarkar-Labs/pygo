package models

import "time"

// GeneratedLlamaV2 records execution state for the llama v2 component.
type GeneratedLlamaV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaV2() *GeneratedLlamaV2 {
	return &GeneratedLlamaV2{Name: "llama_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
