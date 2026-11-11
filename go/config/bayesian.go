package config

import "time"

// GeneratedBayesian records execution state for the bayesian component.
type GeneratedBayesian struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBayesian() *GeneratedBayesian {
	return &GeneratedBayesian{Name: "bayesian", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBayesian) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBayesian) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
