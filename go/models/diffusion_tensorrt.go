package models

import "time"

// GeneratedDiffusionTensorrt records execution state for the diffusion tensorrt component.
type GeneratedDiffusionTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionTensorrt() *GeneratedDiffusionTensorrt {
	return &GeneratedDiffusionTensorrt{Name: "diffusion_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
