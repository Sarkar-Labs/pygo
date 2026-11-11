package deploy

import "time"

// GeneratedTriton records execution state for the triton component.
type GeneratedTriton struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTriton() *GeneratedTriton {
	return &GeneratedTriton{Name: "triton", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTriton) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTriton) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
