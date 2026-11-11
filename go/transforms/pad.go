package transforms

import "time"

// GeneratedPad records execution state for the pad component.
type GeneratedPad struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPad() *GeneratedPad { return &GeneratedPad{Name: "pad", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPad) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPad) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
