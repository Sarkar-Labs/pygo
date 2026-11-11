package models

import "time"

// GeneratedDinoQuantized records execution state for the dino quantized component.
type GeneratedDinoQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoQuantized() *GeneratedDinoQuantized {
	return &GeneratedDinoQuantized{Name: "dino_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
