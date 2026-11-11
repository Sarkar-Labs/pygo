package deploy

import "time"

// GeneratedEks records execution state for the eks component.
type GeneratedEks struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEks() *GeneratedEks { return &GeneratedEks{Name: "eks", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEks) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEks) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
