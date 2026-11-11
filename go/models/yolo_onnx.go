package models

import "time"

// GeneratedYoloOnnx records execution state for the yolo onnx component.
type GeneratedYoloOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloOnnx() *GeneratedYoloOnnx {
	return &GeneratedYoloOnnx{Name: "yolo_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
