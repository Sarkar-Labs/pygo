package models

import "time"

// GeneratedEfficientnetTvm records execution state for the efficientnet tvm component.
type GeneratedEfficientnetTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetTvm() *GeneratedEfficientnetTvm {
	return &GeneratedEfficientnetTvm{Name: "efficientnet_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
