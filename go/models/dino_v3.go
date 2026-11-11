package models

import "time"

// GeneratedDinoV3 records execution state for the dino v3 component.
type GeneratedDinoV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoV3() *GeneratedDinoV3 {
	return &GeneratedDinoV3{Name: "dino_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
