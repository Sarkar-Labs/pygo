package models

import "time"

// GeneratedT5Base records execution state for the t5 base component.
type GeneratedT5Base struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedT5Base() *GeneratedT5Base {
	return &GeneratedT5Base{Name: "t5_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedT5Base) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedT5Base) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
