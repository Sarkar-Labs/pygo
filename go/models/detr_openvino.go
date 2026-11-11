package models

import "time"

// GeneratedDetrOpenvino records execution state for the detr openvino component.
type GeneratedDetrOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrOpenvino() *GeneratedDetrOpenvino {
	return &GeneratedDetrOpenvino{Name: "detr_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
