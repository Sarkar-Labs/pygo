package models

import "time"

// GeneratedDinoTiny records execution state for the dino tiny component.
type GeneratedDinoTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoTiny() *GeneratedDinoTiny {
	return &GeneratedDinoTiny{Name: "dino_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
