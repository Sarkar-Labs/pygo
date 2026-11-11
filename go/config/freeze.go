package config

import "time"

// GeneratedFreeze records execution state for the freeze component.
type GeneratedFreeze struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFreeze() *GeneratedFreeze {
	return &GeneratedFreeze{Name: "freeze", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFreeze) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFreeze) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
