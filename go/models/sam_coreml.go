package models

import "time"

// GeneratedSamCoreml records execution state for the sam coreml component.
type GeneratedSamCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamCoreml() *GeneratedSamCoreml {
	return &GeneratedSamCoreml{Name: "sam_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
