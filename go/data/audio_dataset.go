package data

import "time"

// GeneratedAudioDataset records execution state for the audio dataset component.
type GeneratedAudioDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAudioDataset() *GeneratedAudioDataset {
	return &GeneratedAudioDataset{Name: "audio_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAudioDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAudioDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
