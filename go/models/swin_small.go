package models

import "time"

// GeneratedSwinSmall records execution state for the swin small component.
type GeneratedSwinSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSwinSmall() *GeneratedSwinSmall {
	return &GeneratedSwinSmall{Name: "swin_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSwinSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSwinSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
