package models

import "time"

// GeneratedYoloTensorrt records execution state for the yolo tensorrt component.
type GeneratedYoloTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloTensorrt() *GeneratedYoloTensorrt {
	return &GeneratedYoloTensorrt{Name: "yolo_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
