package models

import "time"

// GeneratedDiffusionBase records execution state for the diffusion base component.
type GeneratedDiffusionBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionBase() *GeneratedDiffusionBase {
	return &GeneratedDiffusionBase{Name: "diffusion_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
