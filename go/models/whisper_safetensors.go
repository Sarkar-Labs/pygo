package models

import "time"

// GeneratedWhisperSafetensors records execution state for the whisper safetensors component.
type GeneratedWhisperSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperSafetensors() *GeneratedWhisperSafetensors {
	return &GeneratedWhisperSafetensors{Name: "whisper_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
