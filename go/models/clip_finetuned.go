package models

import "time"

// GeneratedClipFinetuned records execution state for the clip finetuned component.
type GeneratedClipFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipFinetuned() *GeneratedClipFinetuned {
	return &GeneratedClipFinetuned{Name: "clip_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
