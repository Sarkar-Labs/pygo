package models

import "time"

// GeneratedYoloLarge records execution state for the yolo large component.
type GeneratedYoloLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloLarge() *GeneratedYoloLarge {
	return &GeneratedYoloLarge{Name: "yolo_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
