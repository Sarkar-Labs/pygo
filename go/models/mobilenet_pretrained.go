package models

import "time"

// GeneratedMobilenetPretrained records execution state for the mobilenet pretrained component.
type GeneratedMobilenetPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetPretrained() *GeneratedMobilenetPretrained {
	return &GeneratedMobilenetPretrained{Name: "mobilenet_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
