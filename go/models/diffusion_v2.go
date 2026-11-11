package models

import "time"

// GeneratedDiffusionV2 records execution state for the diffusion v2 component.
type GeneratedDiffusionV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionV2() *GeneratedDiffusionV2 {
	return &GeneratedDiffusionV2{Name: "diffusion_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
