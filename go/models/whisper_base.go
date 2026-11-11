package models

import "time"

// GeneratedWhisperBase records execution state for the whisper base component.
type GeneratedWhisperBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperBase() *GeneratedWhisperBase {
	return &GeneratedWhisperBase{Name: "whisper_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
