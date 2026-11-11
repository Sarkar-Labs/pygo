package models

import "time"

// GeneratedUnetOpenvino records execution state for the unet openvino component.
type GeneratedUnetOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetOpenvino() *GeneratedUnetOpenvino {
	return &GeneratedUnetOpenvino{Name: "unet_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
