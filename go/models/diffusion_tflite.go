package models

import "time"

// GeneratedDiffusionTflite records execution state for the diffusion tflite component.
type GeneratedDiffusionTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionTflite() *GeneratedDiffusionTflite {
	return &GeneratedDiffusionTflite{Name: "diffusion_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
