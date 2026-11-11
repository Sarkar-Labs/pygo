package models

import "time"

// GeneratedMobilenetQuantized records execution state for the mobilenet quantized component.
type GeneratedMobilenetQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetQuantized() *GeneratedMobilenetQuantized {
	return &GeneratedMobilenetQuantized{Name: "mobilenet_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
