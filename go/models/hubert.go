package models

import "time"

// GeneratedHubert records execution state for the hubert component.
type GeneratedHubert struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedHubert() *GeneratedHubert {
	return &GeneratedHubert{Name: "hubert", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedHubert) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedHubert) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
