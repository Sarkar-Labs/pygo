package models

import "time"

// GeneratedEfficientnetPruned records execution state for the efficientnet pruned component.
type GeneratedEfficientnetPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetPruned() *GeneratedEfficientnetPruned {
	return &GeneratedEfficientnetPruned{Name: "efficientnet_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
