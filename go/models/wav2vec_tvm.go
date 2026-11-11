package models

import "time"

// GeneratedWav2vecTvm records execution state for the wav2vec tvm component.
type GeneratedWav2vecTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecTvm() *GeneratedWav2vecTvm {
	return &GeneratedWav2vecTvm{Name: "wav2vec_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
