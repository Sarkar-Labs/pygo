package models

import "time"

// GeneratedResnetV1 records execution state for the resnet v1 component.
type GeneratedResnetV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetV1() *GeneratedResnetV1 {
	return &GeneratedResnetV1{Name: "resnet_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
