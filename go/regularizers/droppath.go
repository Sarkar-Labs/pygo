package regularizers

import "time"

// GeneratedDroppath records execution state for the droppath component.
type GeneratedDroppath struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDroppath() *GeneratedDroppath {
	return &GeneratedDroppath{Name: "droppath", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDroppath) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDroppath) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
