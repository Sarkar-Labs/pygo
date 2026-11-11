package data

import "time"

// GeneratedCollate records execution state for the collate component.
type GeneratedCollate struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCollate() *GeneratedCollate {
	return &GeneratedCollate{Name: "collate", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCollate) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCollate) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
