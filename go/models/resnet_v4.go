package models

import "time"

// GeneratedResnetV4 records execution state for the resnet v4 component.
type GeneratedResnetV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetV4() *GeneratedResnetV4 {
	return &GeneratedResnetV4{Name: "resnet_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
