package models

import "time"

// GeneratedDebertaLarge records execution state for the deberta large component.
type GeneratedDebertaLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDebertaLarge() *GeneratedDebertaLarge {
	return &GeneratedDebertaLarge{Name: "deberta_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDebertaLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDebertaLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
