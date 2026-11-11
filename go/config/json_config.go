package config

import "time"

// GeneratedJsonConfig records execution state for the json config component.
type GeneratedJsonConfig struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedJsonConfig() *GeneratedJsonConfig {
	return &GeneratedJsonConfig{Name: "json_config", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedJsonConfig) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedJsonConfig) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
