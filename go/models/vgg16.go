package models

import "time"

// GeneratedVgg16 records execution state for the vgg16 component.
type GeneratedVgg16 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVgg16() *GeneratedVgg16 {
	return &GeneratedVgg16{Name: "vgg16", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVgg16) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVgg16) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
