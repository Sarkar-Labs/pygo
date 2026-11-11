package inference

import "time"

// GeneratedLookaheadDecode records execution state for the lookahead decode component.
type GeneratedLookaheadDecode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLookaheadDecode() *GeneratedLookaheadDecode {
	return &GeneratedLookaheadDecode{Name: "lookahead_decode", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLookaheadDecode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLookaheadDecode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
