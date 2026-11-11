package models

import "time"

// GeneratedLlamaBase records execution state for the llama base component.
type GeneratedLlamaBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaBase() *GeneratedLlamaBase {
	return &GeneratedLlamaBase{Name: "llama_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
