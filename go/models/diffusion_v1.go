package models

import "time"

// GeneratedDiffusionV1 records execution state for the diffusion v1 component.
type GeneratedDiffusionV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionV1() *GeneratedDiffusionV1 {
	return &GeneratedDiffusionV1{Name: "diffusion_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
