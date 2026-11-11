package models

import "time"

// GeneratedStableDiffusion records execution state for the stable diffusion component.
type GeneratedStableDiffusion struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStableDiffusion() *GeneratedStableDiffusion {
	return &GeneratedStableDiffusion{Name: "stable_diffusion", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStableDiffusion) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStableDiffusion) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
