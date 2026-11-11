package models

import "time"

// GeneratedDiffusionQuantized records execution state for the diffusion quantized component.
type GeneratedDiffusionQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionQuantized() *GeneratedDiffusionQuantized {
	return &GeneratedDiffusionQuantized{Name: "diffusion_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
