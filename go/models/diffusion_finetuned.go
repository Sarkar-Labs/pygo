package models

import "time"

// GeneratedDiffusionFinetuned records execution state for the diffusion finetuned component.
type GeneratedDiffusionFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionFinetuned() *GeneratedDiffusionFinetuned {
	return &GeneratedDiffusionFinetuned{Name: "diffusion_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
