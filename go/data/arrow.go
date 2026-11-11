package data

import "time"

// GeneratedArrow records execution state for the arrow component.
type GeneratedArrow struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedArrow() *GeneratedArrow {
	return &GeneratedArrow{Name: "arrow", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedArrow) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedArrow) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
