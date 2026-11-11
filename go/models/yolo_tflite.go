package models

import "time"

// GeneratedYoloTflite records execution state for the yolo tflite component.
type GeneratedYoloTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloTflite() *GeneratedYoloTflite {
	return &GeneratedYoloTflite{Name: "yolo_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
