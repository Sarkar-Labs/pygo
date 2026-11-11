package models

import "time"

// GeneratedDinoTvm records execution state for the dino tvm component.
type GeneratedDinoTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoTvm() *GeneratedDinoTvm {
	return &GeneratedDinoTvm{Name: "dino_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
