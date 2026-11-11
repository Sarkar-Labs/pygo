package models

import "time"

// GeneratedLlamaV3 records execution state for the llama v3 component.
type GeneratedLlamaV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaV3() *GeneratedLlamaV3 {
	return &GeneratedLlamaV3{Name: "llama_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
