package models

import "time"

// GeneratedYolov8 records execution state for the yolov8 component.
type GeneratedYolov8 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYolov8() *GeneratedYolov8 {
	return &GeneratedYolov8{Name: "yolov8", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYolov8) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYolov8) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
