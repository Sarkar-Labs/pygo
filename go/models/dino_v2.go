package models

import "time"

// GeneratedDinoV2 records execution state for the dino v2 component.
type GeneratedDinoV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoV2() *GeneratedDinoV2 {
	return &GeneratedDinoV2{Name: "dino_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
