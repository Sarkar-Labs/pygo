package models

import "time"

// GeneratedMobilenetTvm records execution state for the mobilenet tvm component.
type GeneratedMobilenetTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetTvm() *GeneratedMobilenetTvm {
	return &GeneratedMobilenetTvm{Name: "mobilenet_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
