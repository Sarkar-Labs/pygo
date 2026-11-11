package models

import "time"

// GeneratedT5Xl records execution state for the t5 xl component.
type GeneratedT5Xl struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedT5Xl() *GeneratedT5Xl { return &GeneratedT5Xl{Name: "t5_xl", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedT5Xl) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedT5Xl) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
