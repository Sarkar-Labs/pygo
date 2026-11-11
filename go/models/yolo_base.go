package models

import "time"

// GeneratedYoloBase records execution state for the yolo base component.
type GeneratedYoloBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloBase() *GeneratedYoloBase {
	return &GeneratedYoloBase{Name: "yolo_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
