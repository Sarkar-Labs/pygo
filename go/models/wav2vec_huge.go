package models

import "time"

// GeneratedWav2vecHuge records execution state for the wav2vec huge component.
type GeneratedWav2vecHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecHuge() *GeneratedWav2vecHuge {
	return &GeneratedWav2vecHuge{Name: "wav2vec_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
