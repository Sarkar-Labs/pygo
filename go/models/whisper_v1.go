package models

import "time"

// GeneratedWhisperV1 records execution state for the whisper v1 component.
type GeneratedWhisperV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperV1() *GeneratedWhisperV1 {
	return &GeneratedWhisperV1{Name: "whisper_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
