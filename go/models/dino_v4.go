package models

import "time"

// GeneratedDinoV4 records execution state for the dino v4 component.
type GeneratedDinoV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoV4() *GeneratedDinoV4 {
	return &GeneratedDinoV4{Name: "dino_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
