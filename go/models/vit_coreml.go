package models

import "time"

// GeneratedVitCoreml records execution state for the vit coreml component.
type GeneratedVitCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitCoreml() *GeneratedVitCoreml {
	return &GeneratedVitCoreml{Name: "vit_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
