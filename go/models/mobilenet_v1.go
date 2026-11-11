package models

import "time"

// GeneratedMobilenetV1 records execution state for the mobilenet v1 component.
type GeneratedMobilenetV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetV1() *GeneratedMobilenetV1 {
	return &GeneratedMobilenetV1{Name: "mobilenet_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
