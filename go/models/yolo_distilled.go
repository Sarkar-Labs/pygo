package models

import "time"

// GeneratedYoloDistilled records execution state for the yolo distilled component.
type GeneratedYoloDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloDistilled() *GeneratedYoloDistilled {
	return &GeneratedYoloDistilled{Name: "yolo_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
