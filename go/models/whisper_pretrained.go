package models

import "time"

// GeneratedWhisperPretrained records execution state for the whisper pretrained component.
type GeneratedWhisperPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperPretrained() *GeneratedWhisperPretrained {
	return &GeneratedWhisperPretrained{Name: "whisper_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
