package config

import "time"

// GeneratedFlags records execution state for the flags component.
type GeneratedFlags struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFlags() *GeneratedFlags {
	return &GeneratedFlags{Name: "flags", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFlags) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFlags) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
