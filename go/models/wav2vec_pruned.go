package models

import "time"

// GeneratedWav2vecPruned records execution state for the wav2vec pruned component.
type GeneratedWav2vecPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecPruned() *GeneratedWav2vecPruned {
	return &GeneratedWav2vecPruned{Name: "wav2vec_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
