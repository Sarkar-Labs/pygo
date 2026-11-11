package models

import "time"

// GeneratedWav2vecPretrained records execution state for the wav2vec pretrained component.
type GeneratedWav2vecPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecPretrained() *GeneratedWav2vecPretrained {
	return &GeneratedWav2vecPretrained{Name: "wav2vec_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
