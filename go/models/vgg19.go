package models

import "time"

// GeneratedVgg19 records execution state for the vgg19 component.
type GeneratedVgg19 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVgg19() *GeneratedVgg19 {
	return &GeneratedVgg19{Name: "vgg19", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVgg19) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVgg19) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
