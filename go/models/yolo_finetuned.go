package models

import "time"

// GeneratedYoloFinetuned records execution state for the yolo finetuned component.
type GeneratedYoloFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloFinetuned() *GeneratedYoloFinetuned {
	return &GeneratedYoloFinetuned{Name: "yolo_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
