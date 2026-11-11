package models

import "time"

// GeneratedWhisperOnnx records execution state for the whisper onnx component.
type GeneratedWhisperOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWhisperOnnx() *GeneratedWhisperOnnx {
	return &GeneratedWhisperOnnx{Name: "whisper_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWhisperOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWhisperOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
