package models

import "time"

// GeneratedDinoBase records execution state for the dino base component.
type GeneratedDinoBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoBase() *GeneratedDinoBase {
	return &GeneratedDinoBase{Name: "dino_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
