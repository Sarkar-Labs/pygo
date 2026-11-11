package inference

import "time"

// GeneratedLengthPenalty records execution state for the length penalty component.
type GeneratedLengthPenalty struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLengthPenalty() *GeneratedLengthPenalty {
	return &GeneratedLengthPenalty{Name: "length_penalty", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLengthPenalty) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLengthPenalty) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
