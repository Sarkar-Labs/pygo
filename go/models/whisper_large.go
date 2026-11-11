package models

import "time"

// GeneratedWhisperLarge records execution state for the whisper large component.
type GeneratedWhisperLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperLarge() *GeneratedWhisperLarge {
	return &GeneratedWhisperLarge{Name: "whisper_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
