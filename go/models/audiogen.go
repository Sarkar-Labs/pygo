package models

import "time"

// GeneratedAudiogen records execution state for the audiogen component.
type GeneratedAudiogen struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAudiogen() *GeneratedAudiogen {
	return &GeneratedAudiogen{Name: "audiogen", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAudiogen) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAudiogen) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
