package models

import "time"

// GeneratedWhisperTensorrt records execution state for the whisper tensorrt component.
type GeneratedWhisperTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperTensorrt() *GeneratedWhisperTensorrt {
	return &GeneratedWhisperTensorrt{Name: "whisper_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
