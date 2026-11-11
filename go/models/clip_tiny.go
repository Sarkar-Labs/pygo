package models

import "time"

// GeneratedClipTiny records execution state for the clip tiny component.
type GeneratedClipTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipTiny() *GeneratedClipTiny {
	return &GeneratedClipTiny{Name: "clip_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
