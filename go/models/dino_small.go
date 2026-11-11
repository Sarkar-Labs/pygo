package models

import "time"

// GeneratedDinoSmall records execution state for the dino small component.
type GeneratedDinoSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoSmall() *GeneratedDinoSmall {
	return &GeneratedDinoSmall{Name: "dino_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
