package models

import "time"

// GeneratedClipDistilled records execution state for the clip distilled component.
type GeneratedClipDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipDistilled() *GeneratedClipDistilled {
	return &GeneratedClipDistilled{Name: "clip_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
