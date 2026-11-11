package models

import "time"

// GeneratedResnetV3 records execution state for the resnet v3 component.
type GeneratedResnetV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetV3() *GeneratedResnetV3 {
	return &GeneratedResnetV3{Name: "resnet_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
