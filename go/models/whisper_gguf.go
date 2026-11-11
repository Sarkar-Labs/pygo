package models

import "time"

// GeneratedWhisperGguf records execution state for the whisper gguf component.
type GeneratedWhisperGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperGguf() *GeneratedWhisperGguf {
	return &GeneratedWhisperGguf{Name: "whisper_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
