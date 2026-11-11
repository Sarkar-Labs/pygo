package models

import "time"

// GeneratedMask2former records execution state for the mask2former component.
type GeneratedMask2former struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMask2former() *GeneratedMask2former {
	return &GeneratedMask2former{Name: "mask2former", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMask2former) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMask2former) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
