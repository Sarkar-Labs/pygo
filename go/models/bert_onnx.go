package models

import "time"

// GeneratedBertOnnx records execution state for the bert onnx component.
type GeneratedBertOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertOnnx() *GeneratedBertOnnx {
	return &GeneratedBertOnnx{Name: "bert_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
