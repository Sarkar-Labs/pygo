package models

import "time"

// GeneratedMobilenetCoreml records execution state for the mobilenet coreml component.
type GeneratedMobilenetCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetCoreml() *GeneratedMobilenetCoreml {
	return &GeneratedMobilenetCoreml{Name: "mobilenet_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
