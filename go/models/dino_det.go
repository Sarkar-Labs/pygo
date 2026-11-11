package models

import "time"

// GeneratedDinoDet records execution state for the dino det component.
type GeneratedDinoDet struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoDet() *GeneratedDinoDet {
	return &GeneratedDinoDet{Name: "dino_det", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoDet) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoDet) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
