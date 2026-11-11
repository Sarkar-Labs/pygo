package inference

import "time"

// GeneratedRepetitionPenalty records execution state for the repetition penalty component.
type GeneratedRepetitionPenalty struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRepetitionPenalty() *GeneratedRepetitionPenalty {
	return &GeneratedRepetitionPenalty{Name: "repetition_penalty", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRepetitionPenalty) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRepetitionPenalty) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
