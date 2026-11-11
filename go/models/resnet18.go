package models

import "time"

// GeneratedResnet18 records execution state for the resnet18 component.
type GeneratedResnet18 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnet18() *GeneratedResnet18 {
	return &GeneratedResnet18{Name: "resnet18", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnet18) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnet18) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
