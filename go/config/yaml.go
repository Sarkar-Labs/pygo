package config

import "time"

// GeneratedYaml records execution state for the yaml component.
type GeneratedYaml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYaml() *GeneratedYaml { return &GeneratedYaml{Name: "yaml", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYaml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYaml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
