package models

import "time"

// GeneratedDinoDistilled records execution state for the dino distilled component.
type GeneratedDinoDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoDistilled() *GeneratedDinoDistilled {
	return &GeneratedDinoDistilled{Name: "dino_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
