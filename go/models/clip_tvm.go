package models

import "time"

// GeneratedClipTvm records execution state for the clip tvm component.
type GeneratedClipTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipTvm() *GeneratedClipTvm {
	return &GeneratedClipTvm{Name: "clip_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
