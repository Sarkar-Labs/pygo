package models

import "time"

// GeneratedBertLarge records execution state for the bert large component.
type GeneratedBertLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertLarge() *GeneratedBertLarge {
	return &GeneratedBertLarge{Name: "bert_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
