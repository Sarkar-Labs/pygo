package models

import "time"

// GeneratedSamQuantized records execution state for the sam quantized component.
type GeneratedSamQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamQuantized() *GeneratedSamQuantized {
	return &GeneratedSamQuantized{Name: "sam_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
