package models

import "time"

// GeneratedUnetFinetuned records execution state for the unet finetuned component.
type GeneratedUnetFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetFinetuned() *GeneratedUnetFinetuned {
	return &GeneratedUnetFinetuned{Name: "unet_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
