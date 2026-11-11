package models

import "time"

// GeneratedSamLarge records execution state for the sam large component.
type GeneratedSamLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamLarge() *GeneratedSamLarge {
	return &GeneratedSamLarge{Name: "sam_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
