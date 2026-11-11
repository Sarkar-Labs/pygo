package models

import "time"

// GeneratedDiffusionDistilled records execution state for the diffusion distilled component.
type GeneratedDiffusionDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionDistilled() *GeneratedDiffusionDistilled {
	return &GeneratedDiffusionDistilled{Name: "diffusion_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
