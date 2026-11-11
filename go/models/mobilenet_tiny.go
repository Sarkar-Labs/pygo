package models

import "time"

// GeneratedMobilenetTiny records execution state for the mobilenet tiny component.
type GeneratedMobilenetTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetTiny() *GeneratedMobilenetTiny {
	return &GeneratedMobilenetTiny{Name: "mobilenet_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
