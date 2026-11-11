package models

import "time"

// GeneratedMobilenetOpenvino records execution state for the mobilenet openvino component.
type GeneratedMobilenetOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetOpenvino() *GeneratedMobilenetOpenvino {
	return &GeneratedMobilenetOpenvino{Name: "mobilenet_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
