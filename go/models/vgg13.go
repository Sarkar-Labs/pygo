package models

import "time"

// GeneratedVgg13 records execution state for the vgg13 component.
type GeneratedVgg13 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVgg13() *GeneratedVgg13 {
	return &GeneratedVgg13{Name: "vgg13", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVgg13) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVgg13) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
