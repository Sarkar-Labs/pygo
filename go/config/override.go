package config

import "time"

// GeneratedOverride records execution state for the override component.
type GeneratedOverride struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOverride() *GeneratedOverride {
	return &GeneratedOverride{Name: "override", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOverride) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOverride) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
