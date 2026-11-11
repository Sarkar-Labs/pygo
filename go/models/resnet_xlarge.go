package models

import "time"

// GeneratedResnetXlarge records execution state for the resnet xlarge component.
type GeneratedResnetXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetXlarge() *GeneratedResnetXlarge {
	return &GeneratedResnetXlarge{Name: "resnet_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
