package config

import "time"

// GeneratedDefaults records execution state for the defaults component.
type GeneratedDefaults struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDefaults() *GeneratedDefaults {
	return &GeneratedDefaults{Name: "defaults", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDefaults) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDefaults) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
