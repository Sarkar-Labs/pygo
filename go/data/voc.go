package data

import "time"

// GeneratedVoc records execution state for the voc component.
type GeneratedVoc struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVoc() *GeneratedVoc { return &GeneratedVoc{Name: "voc", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVoc) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVoc) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
