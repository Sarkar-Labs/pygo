package models

import "time"

// GeneratedSamFinetuned records execution state for the sam finetuned component.
type GeneratedSamFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamFinetuned() *GeneratedSamFinetuned {
	return &GeneratedSamFinetuned{Name: "sam_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
