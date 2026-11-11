package config

import "time"

// GeneratedEnv records execution state for the env component.
type GeneratedEnv struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEnv() *GeneratedEnv { return &GeneratedEnv{Name: "env", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEnv) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEnv) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
