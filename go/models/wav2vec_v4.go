package models

import "time"

// GeneratedWav2vecV4 records execution state for the wav2vec v4 component.
type GeneratedWav2vecV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecV4() *GeneratedWav2vecV4 {
	return &GeneratedWav2vecV4{Name: "wav2vec_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
