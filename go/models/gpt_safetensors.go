package models

import "time"

// GeneratedGptSafetensors records execution state for the gpt safetensors component.
type GeneratedGptSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptSafetensors() *GeneratedGptSafetensors {
	return &GeneratedGptSafetensors{Name: "gpt_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
