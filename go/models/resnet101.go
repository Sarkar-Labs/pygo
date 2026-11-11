package models

import "time"

// GeneratedResnet101 records execution state for the resnet101 component.
type GeneratedResnet101 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnet101() *GeneratedResnet101 {
	return &GeneratedResnet101{Name: "resnet101", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnet101) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnet101) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
