package models

import "time"

// GeneratedResnetQuantized records execution state for the resnet quantized component.
type GeneratedResnetQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetQuantized() *GeneratedResnetQuantized {
	return &GeneratedResnetQuantized{Name: "resnet_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
