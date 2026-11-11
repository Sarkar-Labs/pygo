package models

import "time"

// GeneratedMistralSafetensors records execution state for the mistral safetensors component.
type GeneratedMistralSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralSafetensors() *GeneratedMistralSafetensors {
	return &GeneratedMistralSafetensors{Name: "mistral_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
