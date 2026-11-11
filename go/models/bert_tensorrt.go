package models

import "time"

// GeneratedBertTensorrt records execution state for the bert tensorrt component.
type GeneratedBertTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertTensorrt() *GeneratedBertTensorrt {
	return &GeneratedBertTensorrt{Name: "bert_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
