package models

import "time"

// GeneratedT5Xxl records execution state for the t5 xxl component.
type GeneratedT5Xxl struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedT5Xxl() *GeneratedT5Xxl {
	return &GeneratedT5Xxl{Name: "t5_xxl", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedT5Xxl) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedT5Xxl) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
