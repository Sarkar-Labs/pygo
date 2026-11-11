package models

import "time"

// GeneratedClipOpenvino records execution state for the clip openvino component.
type GeneratedClipOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipOpenvino() *GeneratedClipOpenvino {
	return &GeneratedClipOpenvino{Name: "clip_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
