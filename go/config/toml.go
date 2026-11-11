package config

import "time"

// GeneratedToml records execution state for the toml component.
type GeneratedToml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedToml() *GeneratedToml { return &GeneratedToml{Name: "toml", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedToml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedToml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
