package models

import "time"

// GeneratedSwinBase records execution state for the swin base component.
type GeneratedSwinBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSwinBase() *GeneratedSwinBase {
	return &GeneratedSwinBase{Name: "swin_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSwinBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSwinBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
