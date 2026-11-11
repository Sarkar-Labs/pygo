package models

import "time"

// GeneratedDiffusionSmall records execution state for the diffusion small component.
type GeneratedDiffusionSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionSmall() *GeneratedDiffusionSmall {
	return &GeneratedDiffusionSmall{Name: "diffusion_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
