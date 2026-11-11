package models

import "time"

// GeneratedResnetHuge records execution state for the resnet huge component.
type GeneratedResnetHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetHuge() *GeneratedResnetHuge {
	return &GeneratedResnetHuge{Name: "resnet_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
