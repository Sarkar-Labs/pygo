package models

import "time"

// GeneratedVitOpenvino records execution state for the vit openvino component.
type GeneratedVitOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitOpenvino() *GeneratedVitOpenvino {
	return &GeneratedVitOpenvino{Name: "vit_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
