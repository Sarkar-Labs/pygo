package models

import "time"

// GeneratedMobilenetV4 records execution state for the mobilenet v4 component.
type GeneratedMobilenetV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetV4() *GeneratedMobilenetV4 {
	return &GeneratedMobilenetV4{Name: "mobilenet_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
