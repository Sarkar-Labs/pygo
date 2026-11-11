package models

import "time"

// GeneratedWhisperSmall records execution state for the whisper small component.
type GeneratedWhisperSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperSmall() *GeneratedWhisperSmall {
	return &GeneratedWhisperSmall{Name: "whisper_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
