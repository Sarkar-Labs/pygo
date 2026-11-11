package models

import "time"

// GeneratedDinoPruned records execution state for the dino pruned component.
type GeneratedDinoPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoPruned() *GeneratedDinoPruned {
	return &GeneratedDinoPruned{Name: "dino_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
