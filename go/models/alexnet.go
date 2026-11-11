package models

import "time"

// GeneratedAlexnet records execution state for the alexnet component.
type GeneratedAlexnet struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAlexnet() *GeneratedAlexnet {
	return &GeneratedAlexnet{Name: "alexnet", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAlexnet) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAlexnet) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
