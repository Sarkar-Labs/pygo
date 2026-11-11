package models

import "time"

// GeneratedResnetTiny records execution state for the resnet tiny component.
type GeneratedResnetTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetTiny() *GeneratedResnetTiny {
	return &GeneratedResnetTiny{Name: "resnet_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
