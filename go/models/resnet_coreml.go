package models

import "time"

// GeneratedResnetCoreml records execution state for the resnet coreml component.
type GeneratedResnetCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetCoreml() *GeneratedResnetCoreml {
	return &GeneratedResnetCoreml{Name: "resnet_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
