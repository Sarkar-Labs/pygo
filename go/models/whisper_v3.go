package models

import "time"

// GeneratedWhisperV3 records execution state for the whisper v3 component.
type GeneratedWhisperV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperV3() *GeneratedWhisperV3 {
	return &GeneratedWhisperV3{Name: "whisper_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
