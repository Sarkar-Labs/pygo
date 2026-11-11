package deploy

import "time"

// GeneratedHelm records execution state for the helm component.
type GeneratedHelm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedHelm() *GeneratedHelm { return &GeneratedHelm{Name: "helm", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedHelm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedHelm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
