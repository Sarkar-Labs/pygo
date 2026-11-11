package models

import "time"

// GeneratedWav2vecOnnx records execution state for the wav2vec onnx component.
type GeneratedWav2vecOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecOnnx() *GeneratedWav2vecOnnx {
	return &GeneratedWav2vecOnnx{Name: "wav2vec_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
