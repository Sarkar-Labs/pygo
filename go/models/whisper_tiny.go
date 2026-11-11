package models

import "time"

// GeneratedWhisperTiny records execution state for the whisper tiny component.
type GeneratedWhisperTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperTiny() *GeneratedWhisperTiny {
	return &GeneratedWhisperTiny{Name: "whisper_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
