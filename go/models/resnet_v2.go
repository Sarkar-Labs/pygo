package models

import "time"

// GeneratedResnetV2 records execution state for the resnet v2 component.
type GeneratedResnetV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetV2() *GeneratedResnetV2 {
	return &GeneratedResnetV2{Name: "resnet_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
