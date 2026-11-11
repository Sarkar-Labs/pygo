package models

import "time"

// GeneratedMobilenetDistilled records execution state for the mobilenet distilled component.
type GeneratedMobilenetDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetDistilled() *GeneratedMobilenetDistilled {
	return &GeneratedMobilenetDistilled{Name: "mobilenet_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
