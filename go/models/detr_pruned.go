package models

import "time"

// GeneratedDetrPruned records execution state for the detr pruned component.
type GeneratedDetrPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrPruned() *GeneratedDetrPruned {
	return &GeneratedDetrPruned{Name: "detr_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
