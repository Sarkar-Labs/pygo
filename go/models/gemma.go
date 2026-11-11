package models

import "time"

// GeneratedGemma records execution state for the gemma component.
type GeneratedGemma struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGemma() *GeneratedGemma {
	return &GeneratedGemma{Name: "gemma", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGemma) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGemma) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
