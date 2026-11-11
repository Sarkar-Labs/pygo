package models

import "time"

// GeneratedLlamaQuantized records execution state for the llama quantized component.
type GeneratedLlamaQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaQuantized() *GeneratedLlamaQuantized {
	return &GeneratedLlamaQuantized{Name: "llama_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
