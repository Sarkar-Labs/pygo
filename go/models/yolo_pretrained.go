package models

import "time"

// GeneratedYoloPretrained records execution state for the yolo pretrained component.
type GeneratedYoloPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloPretrained() *GeneratedYoloPretrained {
	return &GeneratedYoloPretrained{Name: "yolo_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
