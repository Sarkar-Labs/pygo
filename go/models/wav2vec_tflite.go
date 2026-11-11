package models

import "time"

// GeneratedWav2vecTflite records execution state for the wav2vec tflite component.
type GeneratedWav2vecTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecTflite() *GeneratedWav2vecTflite {
	return &GeneratedWav2vecTflite{Name: "wav2vec_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
