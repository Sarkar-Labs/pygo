package models

import "time"

// GeneratedMobilenetHuge records execution state for the mobilenet huge component.
type GeneratedMobilenetHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetHuge() *GeneratedMobilenetHuge {
	return &GeneratedMobilenetHuge{Name: "mobilenet_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
