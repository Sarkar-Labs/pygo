package models

import "time"

// GeneratedLlamaGguf records execution state for the llama gguf component.
type GeneratedLlamaGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaGguf() *GeneratedLlamaGguf {
	return &GeneratedLlamaGguf{Name: "llama_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
