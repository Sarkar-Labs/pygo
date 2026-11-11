package models

import "time"

// GeneratedWav2vecCoreml records execution state for the wav2vec coreml component.
type GeneratedWav2vecCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWav2vecCoreml() *GeneratedWav2vecCoreml {
	return &GeneratedWav2vecCoreml{Name: "wav2vec_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWav2vecCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWav2vecCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
