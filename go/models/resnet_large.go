package models

import "time"

// GeneratedResnetLarge records execution state for the resnet large component.
type GeneratedResnetLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetLarge() *GeneratedResnetLarge {
	return &GeneratedResnetLarge{Name: "resnet_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
