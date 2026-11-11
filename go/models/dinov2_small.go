package models

import "time"

// GeneratedDinov2Small records execution state for the dinov2 small component.
type GeneratedDinov2Small struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinov2Small() *GeneratedDinov2Small {
	return &GeneratedDinov2Small{Name: "dinov2_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinov2Small) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinov2Small) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
