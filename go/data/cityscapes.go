package data

import "time"

// GeneratedCityscapes records execution state for the cityscapes component.
type GeneratedCityscapes struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCityscapes() *GeneratedCityscapes {
	return &GeneratedCityscapes{Name: "cityscapes", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCityscapes) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCityscapes) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
