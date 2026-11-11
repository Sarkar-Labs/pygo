package models

import "time"

// GeneratedWav2vecV2 records execution state for the wav2vec v2 component.
type GeneratedWav2vecV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecV2() *GeneratedWav2vecV2 {
	return &GeneratedWav2vecV2{Name: "wav2vec_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
