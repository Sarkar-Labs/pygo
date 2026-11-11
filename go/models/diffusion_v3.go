package models

import "time"

// GeneratedDiffusionV3 records execution state for the diffusion v3 component.
type GeneratedDiffusionV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionV3() *GeneratedDiffusionV3 {
	return &GeneratedDiffusionV3{Name: "diffusion_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
