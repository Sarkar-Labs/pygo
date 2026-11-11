package models

import "time"

// GeneratedYoloOpenvino records execution state for the yolo openvino component.
type GeneratedYoloOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloOpenvino() *GeneratedYoloOpenvino {
	return &GeneratedYoloOpenvino{Name: "yolo_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
