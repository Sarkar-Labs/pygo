package models

import "time"

// GeneratedMobilenetV3Small records execution state for the mobilenet v3 small component.
type GeneratedMobilenetV3Small struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetV3Small() *GeneratedMobilenetV3Small {
	return &GeneratedMobilenetV3Small{Name: "mobilenet_v3_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetV3Small) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetV3Small) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
