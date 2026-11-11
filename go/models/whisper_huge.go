package models

import "time"

// GeneratedWhisperHuge records execution state for the whisper huge component.
type GeneratedWhisperHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperHuge() *GeneratedWhisperHuge {
	return &GeneratedWhisperHuge{Name: "whisper_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
