package models

import "time"

// GeneratedDiffusionXlarge records execution state for the diffusion xlarge component.
type GeneratedDiffusionXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionXlarge() *GeneratedDiffusionXlarge {
	return &GeneratedDiffusionXlarge{Name: "diffusion_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
