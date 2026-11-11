package models

import "time"

// GeneratedDinoFinetuned records execution state for the dino finetuned component.
type GeneratedDinoFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoFinetuned() *GeneratedDinoFinetuned {
	return &GeneratedDinoFinetuned{Name: "dino_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
