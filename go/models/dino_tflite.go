package models

import "time"

// GeneratedDinoTflite records execution state for the dino tflite component.
type GeneratedDinoTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoTflite() *GeneratedDinoTflite {
	return &GeneratedDinoTflite{Name: "dino_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
