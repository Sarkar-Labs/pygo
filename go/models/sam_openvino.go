package models

import "time"

// GeneratedSamOpenvino records execution state for the sam openvino component.
type GeneratedSamOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamOpenvino() *GeneratedSamOpenvino {
	return &GeneratedSamOpenvino{Name: "sam_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
