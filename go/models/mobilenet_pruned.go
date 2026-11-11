package models

import "time"

// GeneratedMobilenetPruned records execution state for the mobilenet pruned component.
type GeneratedMobilenetPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetPruned() *GeneratedMobilenetPruned {
	return &GeneratedMobilenetPruned{Name: "mobilenet_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
