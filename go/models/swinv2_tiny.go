package models

import "time"

// GeneratedSwinv2Tiny records execution state for the swinv2 tiny component.
type GeneratedSwinv2Tiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSwinv2Tiny() *GeneratedSwinv2Tiny {
	return &GeneratedSwinv2Tiny{Name: "swinv2_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSwinv2Tiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSwinv2Tiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
