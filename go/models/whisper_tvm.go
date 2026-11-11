package models

import "time"

// GeneratedWhisperTvm records execution state for the whisper tvm component.
type GeneratedWhisperTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperTvm() *GeneratedWhisperTvm {
	return &GeneratedWhisperTvm{Name: "whisper_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
