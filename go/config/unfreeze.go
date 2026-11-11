package config

import "time"

// GeneratedUnfreeze records execution state for the unfreeze component.
type GeneratedUnfreeze struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnfreeze() *GeneratedUnfreeze {
	return &GeneratedUnfreeze{Name: "unfreeze", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnfreeze) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnfreeze) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
