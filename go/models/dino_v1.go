package models

import "time"

// GeneratedDinoV1 records execution state for the dino v1 component.
type GeneratedDinoV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoV1() *GeneratedDinoV1 {
	return &GeneratedDinoV1{Name: "dino_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
