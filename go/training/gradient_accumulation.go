package training

import "time"

// GeneratedGradientAccumulation records execution state for the gradient accumulation component.
type GeneratedGradientAccumulation struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGradientAccumulation() *GeneratedGradientAccumulation {
	return &GeneratedGradientAccumulation{Name: "gradient_accumulation", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGradientAccumulation) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGradientAccumulation) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
