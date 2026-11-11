package models

import "time"

// GeneratedStarcoder records execution state for the starcoder component.
type GeneratedStarcoder struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStarcoder() *GeneratedStarcoder {
	return &GeneratedStarcoder{Name: "starcoder", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStarcoder) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStarcoder) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
