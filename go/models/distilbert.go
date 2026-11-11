package models

import "time"

// GeneratedDistilbert records execution state for the distilbert component.
type GeneratedDistilbert struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDistilbert() *GeneratedDistilbert {
	return &GeneratedDistilbert{Name: "distilbert", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDistilbert) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDistilbert) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
