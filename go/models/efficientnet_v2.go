package models

import "time"

// GeneratedEfficientnetV2 records execution state for the efficientnet v2 component.
type GeneratedEfficientnetV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetV2() *GeneratedEfficientnetV2 {
	return &GeneratedEfficientnetV2{Name: "efficientnet_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
