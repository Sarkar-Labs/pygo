package models

import "time"

// GeneratedDiffusionOnnx records execution state for the diffusion onnx component.
type GeneratedDiffusionOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionOnnx() *GeneratedDiffusionOnnx {
	return &GeneratedDiffusionOnnx{Name: "diffusion_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
