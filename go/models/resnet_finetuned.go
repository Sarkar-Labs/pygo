package models

import "time"

// GeneratedResnetFinetuned records execution state for the resnet finetuned component.
type GeneratedResnetFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetFinetuned() *GeneratedResnetFinetuned {
	return &GeneratedResnetFinetuned{Name: "resnet_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
