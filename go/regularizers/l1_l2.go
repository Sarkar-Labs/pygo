package regularizers

import "time"

// GeneratedL1L2 records execution state for the l1 l2 component.
type GeneratedL1L2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedL1L2() *GeneratedL1L2 { return &GeneratedL1L2{Name: "l1_l2", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedL1L2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedL1L2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
