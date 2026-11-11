package config

import "time"

// GeneratedHyperparams records execution state for the hyperparams component.
type GeneratedHyperparams struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedHyperparams() *GeneratedHyperparams {
	return &GeneratedHyperparams{Name: "hyperparams", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedHyperparams) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedHyperparams) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
