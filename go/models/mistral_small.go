package models

import "time"

// GeneratedMistralSmall records execution state for the mistral small component.
type GeneratedMistralSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralSmall() *GeneratedMistralSmall {
	return &GeneratedMistralSmall{Name: "mistral_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
