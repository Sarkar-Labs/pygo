package models

import "time"

// GeneratedYolov5 records execution state for the yolov5 component.
type GeneratedYolov5 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYolov5() *GeneratedYolov5 {
	return &GeneratedYolov5{Name: "yolov5", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYolov5) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYolov5) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
