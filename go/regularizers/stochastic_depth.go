package regularizers

import "time"

// GeneratedStochasticDepth records execution state for the stochastic depth component.
type GeneratedStochasticDepth struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStochasticDepth() *GeneratedStochasticDepth {
	return &GeneratedStochasticDepth{Name: "stochastic_depth", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStochasticDepth) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStochasticDepth) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
