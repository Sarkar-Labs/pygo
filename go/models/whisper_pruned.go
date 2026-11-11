package models

import "time"

// GeneratedWhisperPruned records execution state for the whisper pruned component.
type GeneratedWhisperPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperPruned() *GeneratedWhisperPruned {
	return &GeneratedWhisperPruned{Name: "whisper_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
