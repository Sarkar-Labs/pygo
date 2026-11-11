package models

import "time"

// GeneratedDiffusionHuge records execution state for the diffusion huge component.
type GeneratedDiffusionHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionHuge() *GeneratedDiffusionHuge {
	return &GeneratedDiffusionHuge{Name: "diffusion_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
