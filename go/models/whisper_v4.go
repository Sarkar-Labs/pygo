package models

import "time"

// GeneratedWhisperV4 records execution state for the whisper v4 component.
type GeneratedWhisperV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperV4() *GeneratedWhisperV4 {
	return &GeneratedWhisperV4{Name: "whisper_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
