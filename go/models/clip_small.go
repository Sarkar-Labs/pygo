package models

import "time"

// GeneratedClipSmall records execution state for the clip small component.
type GeneratedClipSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipSmall() *GeneratedClipSmall {
	return &GeneratedClipSmall{Name: "clip_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
