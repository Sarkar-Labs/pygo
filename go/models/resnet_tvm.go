package models

import "time"

// GeneratedResnetTvm records execution state for the resnet tvm component.
type GeneratedResnetTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetTvm() *GeneratedResnetTvm {
	return &GeneratedResnetTvm{Name: "resnet_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
