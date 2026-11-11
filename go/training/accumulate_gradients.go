package training

import "time"

// GeneratedAccumulateGradients records execution state for the accumulate gradients component.
type GeneratedAccumulateGradients struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAccumulateGradients() *GeneratedAccumulateGradients {
	return &GeneratedAccumulateGradients{Name: "accumulate_gradients", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAccumulateGradients) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAccumulateGradients) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
