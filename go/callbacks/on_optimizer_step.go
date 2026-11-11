package callbacks

import "time"

// GeneratedOnOptimizerStep records execution state for the on optimizer step component.
type GeneratedOnOptimizerStep struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnOptimizerStep() *GeneratedOnOptimizerStep {
	return &GeneratedOnOptimizerStep{Name: "on_optimizer_step", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnOptimizerStep) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnOptimizerStep) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
