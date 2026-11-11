package models

import "time"

// GeneratedMobilenetFinetuned records execution state for the mobilenet finetuned component.
type GeneratedMobilenetFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetFinetuned() *GeneratedMobilenetFinetuned {
	return &GeneratedMobilenetFinetuned{Name: "mobilenet_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
