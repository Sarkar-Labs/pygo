package models

import "time"

// GeneratedWhisperXlarge records execution state for the whisper xlarge component.
type GeneratedWhisperXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperXlarge() *GeneratedWhisperXlarge {
	return &GeneratedWhisperXlarge{Name: "whisper_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
