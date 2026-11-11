package models

import "time"

// GeneratedDiffusionV4 records execution state for the diffusion v4 component.
type GeneratedDiffusionV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionV4() *GeneratedDiffusionV4 {
	return &GeneratedDiffusionV4{Name: "diffusion_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
