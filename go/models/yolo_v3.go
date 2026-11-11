package models

import "time"

// GeneratedYoloV3 records execution state for the yolo v3 component.
type GeneratedYoloV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloV3() *GeneratedYoloV3 {
	return &GeneratedYoloV3{Name: "yolo_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
