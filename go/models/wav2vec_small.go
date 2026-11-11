package models

import "time"

// GeneratedWav2vecSmall records execution state for the wav2vec small component.
type GeneratedWav2vecSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecSmall() *GeneratedWav2vecSmall {
	return &GeneratedWav2vecSmall{Name: "wav2vec_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
