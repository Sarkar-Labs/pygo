package models

import "time"

// GeneratedDiffusionModel records execution state for the diffusion model component.
type GeneratedDiffusionModel struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionModel() *GeneratedDiffusionModel {
	return &GeneratedDiffusionModel{Name: "diffusion_model", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionModel) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionModel) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
