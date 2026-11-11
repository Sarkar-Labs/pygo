package models

import "time"

// GeneratedWav2vecV3 records execution state for the wav2vec v3 component.
type GeneratedWav2vecV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecV3() *GeneratedWav2vecV3 {
	return &GeneratedWav2vecV3{Name: "wav2vec_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
