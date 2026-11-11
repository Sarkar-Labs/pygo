package models

import "time"

// GeneratedDebertaBase records execution state for the deberta base component.
type GeneratedDebertaBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDebertaBase() *GeneratedDebertaBase {
	return &GeneratedDebertaBase{Name: "deberta_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDebertaBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDebertaBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
