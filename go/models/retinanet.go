package models

import "time"

// GeneratedRetinanet records execution state for the retinanet component.
type GeneratedRetinanet struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRetinanet() *GeneratedRetinanet {
	return &GeneratedRetinanet{Name: "retinanet", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRetinanet) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRetinanet) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
