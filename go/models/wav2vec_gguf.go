package models

import "time"

// GeneratedWav2vecGguf records execution state for the wav2vec gguf component.
type GeneratedWav2vecGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecGguf() *GeneratedWav2vecGguf {
	return &GeneratedWav2vecGguf{Name: "wav2vec_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
