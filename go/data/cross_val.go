package data

import "time"

// GeneratedCrossVal records execution state for the cross val component.
type GeneratedCrossVal struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCrossVal() *GeneratedCrossVal {
	return &GeneratedCrossVal{Name: "cross_val", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCrossVal) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCrossVal) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
