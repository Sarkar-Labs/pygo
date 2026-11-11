package models

import "time"

// GeneratedDiffusionPruned records execution state for the diffusion pruned component.
type GeneratedDiffusionPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionPruned() *GeneratedDiffusionPruned {
	return &GeneratedDiffusionPruned{Name: "diffusion_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
