package models

import "time"

// GeneratedLlamaV4 records execution state for the llama v4 component.
type GeneratedLlamaV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaV4() *GeneratedLlamaV4 {
	return &GeneratedLlamaV4{Name: "llama_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
