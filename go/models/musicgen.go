package models

import "time"

// GeneratedMusicgen records execution state for the musicgen component.
type GeneratedMusicgen struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMusicgen() *GeneratedMusicgen {
	return &GeneratedMusicgen{Name: "musicgen", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMusicgen) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMusicgen) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
