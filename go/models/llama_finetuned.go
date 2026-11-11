package models

import "time"

// GeneratedLlamaFinetuned records execution state for the llama finetuned component.
type GeneratedLlamaFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaFinetuned() *GeneratedLlamaFinetuned {
	return &GeneratedLlamaFinetuned{Name: "llama_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
