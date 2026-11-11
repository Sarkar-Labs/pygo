package models

import "time"

// GeneratedVitPruned records execution state for the vit pruned component.
type GeneratedVitPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitPruned() *GeneratedVitPruned {
	return &GeneratedVitPruned{Name: "vit_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
