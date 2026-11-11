package models

import "time"

// GeneratedRobertaBase records execution state for the roberta base component.
type GeneratedRobertaBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRobertaBase() *GeneratedRobertaBase {
	return &GeneratedRobertaBase{Name: "roberta_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRobertaBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRobertaBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
