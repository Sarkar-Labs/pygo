package profiler

import "time"

// GeneratedFlopCounter records execution state for the flop counter component.
type GeneratedFlopCounter struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFlopCounter() *GeneratedFlopCounter {
	return &GeneratedFlopCounter{Name: "flop_counter", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFlopCounter) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFlopCounter) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
