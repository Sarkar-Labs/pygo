package models

import "time"

// GeneratedT5Small records execution state for the t5 small component.
type GeneratedT5Small struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedT5Small() *GeneratedT5Small {
	return &GeneratedT5Small{Name: "t5_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedT5Small) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedT5Small) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
