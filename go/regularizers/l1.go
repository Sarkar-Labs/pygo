package regularizers

import "time"

// GeneratedL1 records execution state for the l1 component.
type GeneratedL1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedL1() *GeneratedL1 { return &GeneratedL1{Name: "l1", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedL1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedL1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
