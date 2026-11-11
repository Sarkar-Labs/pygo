package models

import "time"

// GeneratedEfficientnetOpenvino records execution state for the efficientnet openvino component.
type GeneratedEfficientnetOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetOpenvino() *GeneratedEfficientnetOpenvino {
	return &GeneratedEfficientnetOpenvino{Name: "efficientnet_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
