package models

import "time"

// GeneratedVitQuantized records execution state for the vit quantized component.
type GeneratedVitQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitQuantized() *GeneratedVitQuantized {
	return &GeneratedVitQuantized{Name: "vit_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
