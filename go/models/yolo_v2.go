package models

import "time"

// GeneratedYoloV2 records execution state for the yolo v2 component.
type GeneratedYoloV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloV2() *GeneratedYoloV2 {
	return &GeneratedYoloV2{Name: "yolo_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
