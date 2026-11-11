package models

import "time"

// GeneratedBertPruned records execution state for the bert pruned component.
type GeneratedBertPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertPruned() *GeneratedBertPruned {
	return &GeneratedBertPruned{Name: "bert_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
