package models

import "time"

// GeneratedWhisperDistilled records execution state for the whisper distilled component.
type GeneratedWhisperDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperDistilled() *GeneratedWhisperDistilled {
	return &GeneratedWhisperDistilled{Name: "whisper_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
