package models

import "time"

// GeneratedWhisperOpenvino records execution state for the whisper openvino component.
type GeneratedWhisperOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperOpenvino() *GeneratedWhisperOpenvino {
	return &GeneratedWhisperOpenvino{Name: "whisper_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
