package models

import "time"

// GeneratedDiffusionTiny records execution state for the diffusion tiny component.
type GeneratedDiffusionTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionTiny() *GeneratedDiffusionTiny {
	return &GeneratedDiffusionTiny{Name: "diffusion_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
