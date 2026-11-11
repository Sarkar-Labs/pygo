package models

import "time"

// GeneratedSwinLarge records execution state for the swin large component.
type GeneratedSwinLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSwinLarge() *GeneratedSwinLarge {
	return &GeneratedSwinLarge{Name: "swin_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSwinLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSwinLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
