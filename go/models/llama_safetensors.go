package models

import "time"

// GeneratedLlamaSafetensors records execution state for the llama safetensors component.
type GeneratedLlamaSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaSafetensors() *GeneratedLlamaSafetensors {
	return &GeneratedLlamaSafetensors{Name: "llama_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
