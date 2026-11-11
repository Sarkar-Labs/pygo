package regularizers

import "time"

// GeneratedL2 records execution state for the l2 component.
type GeneratedL2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedL2() *GeneratedL2 { return &GeneratedL2{Name: "l2", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedL2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedL2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
