package models

import "time"

// GeneratedMobilenetV2 records execution state for the mobilenet v2 component.
type GeneratedMobilenetV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetV2() *GeneratedMobilenetV2 {
	return &GeneratedMobilenetV2{Name: "mobilenet_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
