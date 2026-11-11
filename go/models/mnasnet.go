package models

import "time"

// GeneratedMnasnet records execution state for the mnasnet component.
type GeneratedMnasnet struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMnasnet() *GeneratedMnasnet {
	return &GeneratedMnasnet{Name: "mnasnet", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMnasnet) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMnasnet) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
