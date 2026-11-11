package models

import "time"

// GeneratedWav2vecBase records execution state for the wav2vec base component.
type GeneratedWav2vecBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecBase() *GeneratedWav2vecBase {
	return &GeneratedWav2vecBase{Name: "wav2vec_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
