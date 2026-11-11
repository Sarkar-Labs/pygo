package models

import "time"

// GeneratedResnet152 records execution state for the resnet152 component.
type GeneratedResnet152 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnet152() *GeneratedResnet152 {
	return &GeneratedResnet152{Name: "resnet152", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnet152) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnet152) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
