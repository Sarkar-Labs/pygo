package models

import "time"

// GeneratedWhisperFinetuned records execution state for the whisper finetuned component.
type GeneratedWhisperFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperFinetuned() *GeneratedWhisperFinetuned {
	return &GeneratedWhisperFinetuned{Name: "whisper_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
