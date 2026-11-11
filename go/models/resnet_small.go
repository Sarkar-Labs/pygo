package models

import "time"

// GeneratedResnetSmall records execution state for the resnet small component.
type GeneratedResnetSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetSmall() *GeneratedResnetSmall {
	return &GeneratedResnetSmall{Name: "resnet_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
