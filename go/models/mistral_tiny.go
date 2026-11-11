package models

import "time"

// GeneratedMistralTiny records execution state for the mistral tiny component.
type GeneratedMistralTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralTiny() *GeneratedMistralTiny {
	return &GeneratedMistralTiny{Name: "mistral_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
