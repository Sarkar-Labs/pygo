package models

import "time"

// GeneratedResnetPruned records execution state for the resnet pruned component.
type GeneratedResnetPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetPruned() *GeneratedResnetPruned {
	return &GeneratedResnetPruned{Name: "resnet_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
