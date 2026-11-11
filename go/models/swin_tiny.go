package models

import "time"

// GeneratedSwinTiny records execution state for the swin tiny component.
type GeneratedSwinTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSwinTiny() *GeneratedSwinTiny {
	return &GeneratedSwinTiny{Name: "swin_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSwinTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSwinTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
