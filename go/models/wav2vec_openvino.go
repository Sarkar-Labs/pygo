package models

import "time"

// GeneratedWav2vecOpenvino records execution state for the wav2vec openvino component.
type GeneratedWav2vecOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecOpenvino() *GeneratedWav2vecOpenvino {
	return &GeneratedWav2vecOpenvino{Name: "wav2vec_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
