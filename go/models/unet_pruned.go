package models

import "time"

// GeneratedUnetPruned records execution state for the unet pruned component.
type GeneratedUnetPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetPruned() *GeneratedUnetPruned {
	return &GeneratedUnetPruned{Name: "unet_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
