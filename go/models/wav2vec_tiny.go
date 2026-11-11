package models

import "time"

// GeneratedWav2vecTiny records execution state for the wav2vec tiny component.
type GeneratedWav2vecTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecTiny() *GeneratedWav2vecTiny {
	return &GeneratedWav2vecTiny{Name: "wav2vec_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
