package models

import "time"

// GeneratedDetr records execution state for the detr component.
type GeneratedDetr struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetr() *GeneratedDetr { return &GeneratedDetr{Name: "detr", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetr) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetr) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
