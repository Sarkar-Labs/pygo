package models

import "time"

// GeneratedRegnetX records execution state for the regnet x component.
type GeneratedRegnetX struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRegnetX() *GeneratedRegnetX {
	return &GeneratedRegnetX{Name: "regnet_x", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRegnetX) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRegnetX) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
