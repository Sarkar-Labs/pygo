package models

import "time"

// GeneratedWhisperTflite records execution state for the whisper tflite component.
type GeneratedWhisperTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperTflite() *GeneratedWhisperTflite {
	return &GeneratedWhisperTflite{Name: "whisper_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
