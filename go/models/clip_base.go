package models

import "time"

// GeneratedClipBase records execution state for the clip base component.
type GeneratedClipBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipBase() *GeneratedClipBase {
	return &GeneratedClipBase{Name: "clip_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
