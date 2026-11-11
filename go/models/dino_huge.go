package models

import "time"

// GeneratedDinoHuge records execution state for the dino huge component.
type GeneratedDinoHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoHuge() *GeneratedDinoHuge {
	return &GeneratedDinoHuge{Name: "dino_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
