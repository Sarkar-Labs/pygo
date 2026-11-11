package models

import "time"

// GeneratedBertOpenvino records execution state for the bert openvino component.
type GeneratedBertOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertOpenvino() *GeneratedBertOpenvino {
	return &GeneratedBertOpenvino{Name: "bert_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
