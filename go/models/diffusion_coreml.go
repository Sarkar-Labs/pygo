package models

import "time"

// GeneratedDiffusionCoreml records execution state for the diffusion coreml component.
type GeneratedDiffusionCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionCoreml() *GeneratedDiffusionCoreml {
	return &GeneratedDiffusionCoreml{Name: "diffusion_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
