package models

import "time"

// GeneratedDiffusionSafetensors records execution state for the diffusion safetensors component.
type GeneratedDiffusionSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionSafetensors() *GeneratedDiffusionSafetensors {
	return &GeneratedDiffusionSafetensors{Name: "diffusion_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
