package models

import "time"

// GeneratedResnetOpenvino records execution state for the resnet openvino component.
type GeneratedResnetOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetOpenvino() *GeneratedResnetOpenvino {
	return &GeneratedResnetOpenvino{Name: "resnet_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
