package models

import "time"

// GeneratedEfficientnetv2M records execution state for the efficientnetv2 m component.
type GeneratedEfficientnetv2M struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetv2M() *GeneratedEfficientnetv2M {
	return &GeneratedEfficientnetv2M{Name: "efficientnetv2_m", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetv2M) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetv2M) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
