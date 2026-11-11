package models

import "time"

// GeneratedWav2vecV1 records execution state for the wav2vec v1 component.
type GeneratedWav2vecV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecV1() *GeneratedWav2vecV1 {
	return &GeneratedWav2vecV1{Name: "wav2vec_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
