package models

import "time"

// GeneratedSamPruned records execution state for the sam pruned component.
type GeneratedSamPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamPruned() *GeneratedSamPruned {
	return &GeneratedSamPruned{Name: "sam_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
