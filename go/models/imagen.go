package models

import "time"

// GeneratedImagen records execution state for the imagen component.
type GeneratedImagen struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedImagen() *GeneratedImagen {
	return &GeneratedImagen{Name: "imagen", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedImagen) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedImagen) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
