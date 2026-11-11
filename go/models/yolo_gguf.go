package models

import "time"

// GeneratedYoloGguf records execution state for the yolo gguf component.
type GeneratedYoloGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloGguf() *GeneratedYoloGguf {
	return &GeneratedYoloGguf{Name: "yolo_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
