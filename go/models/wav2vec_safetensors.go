package models

import "time"

// GeneratedWav2vecSafetensors records execution state for the wav2vec safetensors component.
type GeneratedWav2vecSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecSafetensors() *GeneratedWav2vecSafetensors {
	return &GeneratedWav2vecSafetensors{Name: "wav2vec_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
