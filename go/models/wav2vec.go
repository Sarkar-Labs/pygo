package models

import "time"

// GeneratedWav2vec records execution state for the wav2vec component.
type GeneratedWav2vec struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vec() *GeneratedWav2vec {
	return &GeneratedWav2vec{Name: "wav2vec", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vec) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vec) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
