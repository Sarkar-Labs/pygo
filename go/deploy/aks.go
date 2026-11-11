package deploy

import "time"

// GeneratedAks records execution state for the aks component.
type GeneratedAks struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAks() *GeneratedAks { return &GeneratedAks{Name: "aks", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAks) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAks) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
