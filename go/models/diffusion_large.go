package models

import "time"

// GeneratedDiffusionLarge records execution state for the diffusion large component.
type GeneratedDiffusionLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionLarge() *GeneratedDiffusionLarge {
	return &GeneratedDiffusionLarge{Name: "diffusion_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
