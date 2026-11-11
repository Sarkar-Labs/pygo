package models

import "time"

// GeneratedDiffusionPretrained records execution state for the diffusion pretrained component.
type GeneratedDiffusionPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionPretrained() *GeneratedDiffusionPretrained {
	return &GeneratedDiffusionPretrained{Name: "diffusion_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
