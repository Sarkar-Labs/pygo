package models

import "time"

// GeneratedYoloV1 records execution state for the yolo v1 component.
type GeneratedYoloV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloV1() *GeneratedYoloV1 {
	return &GeneratedYoloV1{Name: "yolo_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
