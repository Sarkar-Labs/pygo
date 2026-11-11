package models

import "time"

// GeneratedBartLarge records execution state for the bart large component.
type GeneratedBartLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBartLarge() *GeneratedBartLarge {
	return &GeneratedBartLarge{Name: "bart_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBartLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBartLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
