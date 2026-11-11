package models

import "time"

// GeneratedDinoXlarge records execution state for the dino xlarge component.
type GeneratedDinoXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoXlarge() *GeneratedDinoXlarge {
	return &GeneratedDinoXlarge{Name: "dino_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
