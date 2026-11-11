package models

import "time"

// GeneratedRobertaLarge records execution state for the roberta large component.
type GeneratedRobertaLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRobertaLarge() *GeneratedRobertaLarge {
	return &GeneratedRobertaLarge{Name: "roberta_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRobertaLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRobertaLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
