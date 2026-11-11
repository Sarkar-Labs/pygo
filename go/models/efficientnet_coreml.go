package models

import "time"

// GeneratedEfficientnetCoreml records execution state for the efficientnet coreml component.
type GeneratedEfficientnetCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetCoreml() *GeneratedEfficientnetCoreml {
	return &GeneratedEfficientnetCoreml{Name: "efficientnet_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
