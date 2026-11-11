package models

import "time"

// GeneratedYoloV4 records execution state for the yolo v4 component.
type GeneratedYoloV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloV4() *GeneratedYoloV4 {
	return &GeneratedYoloV4{Name: "yolo_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
