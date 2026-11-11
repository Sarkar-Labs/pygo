package models

import "time"

// GeneratedYoloSmall records execution state for the yolo small component.
type GeneratedYoloSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloSmall() *GeneratedYoloSmall {
	return &GeneratedYoloSmall{Name: "yolo_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
