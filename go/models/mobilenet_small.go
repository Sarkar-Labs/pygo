package models

import "time"

// GeneratedMobilenetSmall records execution state for the mobilenet small component.
type GeneratedMobilenetSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetSmall() *GeneratedMobilenetSmall {
	return &GeneratedMobilenetSmall{Name: "mobilenet_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
