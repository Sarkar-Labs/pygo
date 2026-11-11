package models

import "time"

// GeneratedDinoGguf records execution state for the dino gguf component.
type GeneratedDinoGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoGguf() *GeneratedDinoGguf {
	return &GeneratedDinoGguf{Name: "dino_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
