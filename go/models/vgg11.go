package models

import "time"

// GeneratedVgg11 records execution state for the vgg11 component.
type GeneratedVgg11 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVgg11() *GeneratedVgg11 {
	return &GeneratedVgg11{Name: "vgg11", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVgg11) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVgg11) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
