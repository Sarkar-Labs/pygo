package models

import "time"

// GeneratedLlamaV1 records execution state for the llama v1 component.
type GeneratedLlamaV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaV1() *GeneratedLlamaV1 {
	return &GeneratedLlamaV1{Name: "llama_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
