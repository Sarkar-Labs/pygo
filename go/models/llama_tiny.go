package models

import "time"

// GeneratedLlamaTiny records execution state for the llama tiny component.
type GeneratedLlamaTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaTiny() *GeneratedLlamaTiny {
	return &GeneratedLlamaTiny{Name: "llama_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
