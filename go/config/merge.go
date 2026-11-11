package config

import "time"

// GeneratedMerge records execution state for the merge component.
type GeneratedMerge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMerge() *GeneratedMerge {
	return &GeneratedMerge{Name: "merge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMerge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMerge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
