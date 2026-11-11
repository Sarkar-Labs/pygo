package models

import "time"

// GeneratedOpusMt records execution state for the opus mt component.
type GeneratedOpusMt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOpusMt() *GeneratedOpusMt {
	return &GeneratedOpusMt{Name: "opus_mt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOpusMt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOpusMt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
