package autograd

import "time"

// GeneratedVmap records execution state for the vmap component.
type GeneratedVmap struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVmap() *GeneratedVmap { return &GeneratedVmap{Name: "vmap", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVmap) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVmap) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
