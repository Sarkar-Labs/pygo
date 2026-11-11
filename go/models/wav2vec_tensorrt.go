package models

import "time"

// GeneratedWav2vecTensorrt records execution state for the wav2vec tensorrt component.
type GeneratedWav2vecTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecTensorrt() *GeneratedWav2vecTensorrt {
	return &GeneratedWav2vecTensorrt{Name: "wav2vec_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
