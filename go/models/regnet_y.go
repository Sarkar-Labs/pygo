package models

import "time"

// GeneratedRegnetY records execution state for the regnet y component.
type GeneratedRegnetY struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRegnetY() *GeneratedRegnetY {
	return &GeneratedRegnetY{Name: "regnet_y", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRegnetY) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRegnetY) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
