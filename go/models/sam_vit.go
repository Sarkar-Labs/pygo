package models

import "time"

// GeneratedSamVit records execution state for the sam vit component.
type GeneratedSamVit struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamVit() *GeneratedSamVit {
	return &GeneratedSamVit{Name: "sam_vit", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamVit) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamVit) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
