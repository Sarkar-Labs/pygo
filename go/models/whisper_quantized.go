package models

import "time"

// GeneratedWhisperQuantized records execution state for the whisper quantized component.
type GeneratedWhisperQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperQuantized() *GeneratedWhisperQuantized {
	return &GeneratedWhisperQuantized{Name: "whisper_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
