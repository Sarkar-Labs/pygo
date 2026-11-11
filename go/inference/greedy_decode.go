package inference

import "time"

// GeneratedGreedyDecode records execution state for the greedy decode component.
type GeneratedGreedyDecode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGreedyDecode() *GeneratedGreedyDecode {
	return &GeneratedGreedyDecode{Name: "greedy_decode", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGreedyDecode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGreedyDecode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
