package models

import "time"

// GeneratedClipPruned records execution state for the clip pruned component.
type GeneratedClipPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipPruned() *GeneratedClipPruned {
	return &GeneratedClipPruned{Name: "clip_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
