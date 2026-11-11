package models

import "time"

// GeneratedEfficientnetSmall records execution state for the efficientnet small component.
type GeneratedEfficientnetSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetSmall() *GeneratedEfficientnetSmall {
	return &GeneratedEfficientnetSmall{Name: "efficientnet_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
