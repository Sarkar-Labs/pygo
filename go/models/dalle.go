package models

import "time"

// GeneratedDalle records execution state for the dalle component.
type GeneratedDalle struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDalle() *GeneratedDalle {
	return &GeneratedDalle{Name: "dalle", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDalle) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDalle) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
