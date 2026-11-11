package models

import "time"

// GeneratedGptFinetuned records execution state for the gpt finetuned component.
type GeneratedGptFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptFinetuned() *GeneratedGptFinetuned {
	return &GeneratedGptFinetuned{Name: "gpt_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
