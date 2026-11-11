package models

import "time"

// GeneratedElectraLarge records execution state for the electra large component.
type GeneratedElectraLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedElectraLarge() *GeneratedElectraLarge {
	return &GeneratedElectraLarge{Name: "electra_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedElectraLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedElectraLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
