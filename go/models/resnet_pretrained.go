package models

import "time"

// GeneratedResnetPretrained records execution state for the resnet pretrained component.
type GeneratedResnetPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetPretrained() *GeneratedResnetPretrained {
	return &GeneratedResnetPretrained{Name: "resnet_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
