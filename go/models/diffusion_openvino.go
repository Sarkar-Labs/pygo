package models

import "time"

// GeneratedDiffusionOpenvino records execution state for the diffusion openvino component.
type GeneratedDiffusionOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionOpenvino() *GeneratedDiffusionOpenvino {
	return &GeneratedDiffusionOpenvino{Name: "diffusion_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
