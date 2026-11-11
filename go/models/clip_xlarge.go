package models

import "time"

// GeneratedClipXlarge records execution state for the clip xlarge component.
type GeneratedClipXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipXlarge() *GeneratedClipXlarge {
	return &GeneratedClipXlarge{Name: "clip_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
