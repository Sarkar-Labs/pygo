package models

import "time"

// GeneratedWideResnet50 records execution state for the wide resnet50 component.
type GeneratedWideResnet50 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWideResnet50() *GeneratedWideResnet50 {
	return &GeneratedWideResnet50{Name: "wide_resnet50", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWideResnet50) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWideResnet50) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
