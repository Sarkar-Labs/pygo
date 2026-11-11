package models

import "time"

// GeneratedDinoLarge records execution state for the dino large component.
type GeneratedDinoLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoLarge() *GeneratedDinoLarge {
	return &GeneratedDinoLarge{Name: "dino_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
