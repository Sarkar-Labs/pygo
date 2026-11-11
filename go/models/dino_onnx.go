package models

import "time"

// GeneratedDinoOnnx records execution state for the dino onnx component.
type GeneratedDinoOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoOnnx() *GeneratedDinoOnnx {
	return &GeneratedDinoOnnx{Name: "dino_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
