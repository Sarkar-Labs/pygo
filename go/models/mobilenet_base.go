package models

import "time"

// GeneratedMobilenetBase records execution state for the mobilenet base component.
type GeneratedMobilenetBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetBase() *GeneratedMobilenetBase {
	return &GeneratedMobilenetBase{Name: "mobilenet_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
