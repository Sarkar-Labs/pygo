package models

import "time"

// GeneratedDinoOpenvino records execution state for the dino openvino component.
type GeneratedDinoOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoOpenvino() *GeneratedDinoOpenvino {
	return &GeneratedDinoOpenvino{Name: "dino_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
