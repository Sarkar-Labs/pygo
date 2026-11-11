package models

import "time"

// GeneratedXlnetLarge records execution state for the xlnet large component.
type GeneratedXlnetLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedXlnetLarge() *GeneratedXlnetLarge {
	return &GeneratedXlnetLarge{Name: "xlnet_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedXlnetLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedXlnetLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
