package models

import "time"

// GeneratedYoloTiny records execution state for the yolo tiny component.
type GeneratedYoloTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloTiny() *GeneratedYoloTiny {
	return &GeneratedYoloTiny{Name: "yolo_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
