package models

import "time"

// GeneratedEfficientnetTiny records execution state for the efficientnet tiny component.
type GeneratedEfficientnetTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetTiny() *GeneratedEfficientnetTiny {
	return &GeneratedEfficientnetTiny{Name: "efficientnet_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
