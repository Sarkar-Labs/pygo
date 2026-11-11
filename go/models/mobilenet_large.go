package models

import "time"

// GeneratedMobilenetLarge records execution state for the mobilenet large component.
type GeneratedMobilenetLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetLarge() *GeneratedMobilenetLarge {
	return &GeneratedMobilenetLarge{Name: "mobilenet_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
