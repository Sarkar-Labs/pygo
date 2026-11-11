package models

import "time"

// GeneratedAlbertBase records execution state for the albert base component.
type GeneratedAlbertBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAlbertBase() *GeneratedAlbertBase {
	return &GeneratedAlbertBase{Name: "albert_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAlbertBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAlbertBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
