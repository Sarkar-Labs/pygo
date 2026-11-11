package models

import "time"

// GeneratedSwinv2Small records execution state for the swinv2 small component.
type GeneratedSwinv2Small struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSwinv2Small() *GeneratedSwinv2Small {
	return &GeneratedSwinv2Small{Name: "swinv2_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSwinv2Small) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSwinv2Small) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
