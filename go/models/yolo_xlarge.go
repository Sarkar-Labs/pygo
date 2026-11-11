package models

import "time"

// GeneratedYoloXlarge records execution state for the yolo xlarge component.
type GeneratedYoloXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloXlarge() *GeneratedYoloXlarge {
	return &GeneratedYoloXlarge{Name: "yolo_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
