package models

import "time"

// GeneratedDiffusionTvm records execution state for the diffusion tvm component.
type GeneratedDiffusionTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionTvm() *GeneratedDiffusionTvm {
	return &GeneratedDiffusionTvm{Name: "diffusion_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
