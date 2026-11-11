package training

import "time"

// GeneratedZeroOptimizer records execution state for the zero optimizer component.
type GeneratedZeroOptimizer struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedZeroOptimizer() *GeneratedZeroOptimizer {
	return &GeneratedZeroOptimizer{Name: "zero_optimizer", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedZeroOptimizer) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedZeroOptimizer) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
