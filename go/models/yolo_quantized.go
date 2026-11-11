package models

import "time"

// GeneratedYoloQuantized records execution state for the yolo quantized component.
type GeneratedYoloQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloQuantized() *GeneratedYoloQuantized {
	return &GeneratedYoloQuantized{Name: "yolo_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
