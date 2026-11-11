package models

import "time"

// GeneratedGptPruned records execution state for the gpt pruned component.
type GeneratedGptPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptPruned() *GeneratedGptPruned {
	return &GeneratedGptPruned{Name: "gpt_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
