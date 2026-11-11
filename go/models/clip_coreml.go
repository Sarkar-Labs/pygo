package models

import "time"

// GeneratedClipCoreml records execution state for the clip coreml component.
type GeneratedClipCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipCoreml() *GeneratedClipCoreml {
	return &GeneratedClipCoreml{Name: "clip_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
