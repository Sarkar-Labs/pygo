package autograd

import "time"

// GeneratedTape records execution state for the tape component.
type GeneratedTape struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTape() *GeneratedTape { return &GeneratedTape{Name: "tape", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTape) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTape) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
