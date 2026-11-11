package models

import "time"

// GeneratedT5Large records execution state for the t5 large component.
type GeneratedT5Large struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedT5Large() *GeneratedT5Large {
	return &GeneratedT5Large{Name: "t5_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedT5Large) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedT5Large) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
