package models

import "time"

// GeneratedSamOnnx records execution state for the sam onnx component.
type GeneratedSamOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamOnnx() *GeneratedSamOnnx {
	return &GeneratedSamOnnx{Name: "sam_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
