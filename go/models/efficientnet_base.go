package models

import "time"

// GeneratedEfficientnetBase records execution state for the efficientnet base component.
type GeneratedEfficientnetBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetBase() *GeneratedEfficientnetBase {
	return &GeneratedEfficientnetBase{Name: "efficientnet_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
