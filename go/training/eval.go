package training

import "time"

// GeneratedEval records execution state for the eval component.
type GeneratedEval struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEval() *GeneratedEval { return &GeneratedEval{Name: "eval", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEval) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEval) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
