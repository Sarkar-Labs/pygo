package transforms

import "time"

// GeneratedPadSequence records execution state for the pad sequence component.
type GeneratedPadSequence struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPadSequence() *GeneratedPadSequence {
	return &GeneratedPadSequence{Name: "pad_sequence", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPadSequence) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPadSequence) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
