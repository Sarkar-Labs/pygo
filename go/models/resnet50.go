package models

import "time"

// GeneratedResnet50 records execution state for the resnet50 component.
type GeneratedResnet50 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnet50() *GeneratedResnet50 {
	return &GeneratedResnet50{Name: "resnet50", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnet50) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnet50) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
