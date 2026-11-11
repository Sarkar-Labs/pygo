package models

import "time"

// GeneratedMistralGguf records execution state for the mistral gguf component.
type GeneratedMistralGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralGguf() *GeneratedMistralGguf {
	return &GeneratedMistralGguf{Name: "mistral_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
