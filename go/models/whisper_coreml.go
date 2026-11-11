package models

import "time"

// GeneratedWhisperCoreml records execution state for the whisper coreml component.
type GeneratedWhisperCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperCoreml() *GeneratedWhisperCoreml {
	return &GeneratedWhisperCoreml{Name: "whisper_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
