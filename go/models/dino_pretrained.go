package models

import "time"

// GeneratedDinoPretrained records execution state for the dino pretrained component.
type GeneratedDinoPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoPretrained() *GeneratedDinoPretrained {
	return &GeneratedDinoPretrained{Name: "dino_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
