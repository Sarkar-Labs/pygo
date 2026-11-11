package training

import "time"

// GeneratedStep records execution state for the step component.
type GeneratedStep struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStep() *GeneratedStep { return &GeneratedStep{Name: "step", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStep) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStep) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
