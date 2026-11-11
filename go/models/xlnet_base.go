package models

import "time"

// GeneratedXlnetBase records execution state for the xlnet base component.
type GeneratedXlnetBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedXlnetBase() *GeneratedXlnetBase {
	return &GeneratedXlnetBase{Name: "xlnet_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedXlnetBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedXlnetBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
