package models

import "time"

// GeneratedWhisperV2 records execution state for the whisper v2 component.
type GeneratedWhisperV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperV2() *GeneratedWhisperV2 {
	return &GeneratedWhisperV2{Name: "whisper_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
