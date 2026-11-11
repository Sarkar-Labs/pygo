package models

import "time"

// GeneratedMobilenetV3 records execution state for the mobilenet v3 component.
type GeneratedMobilenetV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetV3() *GeneratedMobilenetV3 {
	return &GeneratedMobilenetV3{Name: "mobilenet_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
