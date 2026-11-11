package models

import "time"

// GeneratedBertTflite records execution state for the bert tflite component.
type GeneratedBertTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertTflite() *GeneratedBertTflite {
	return &GeneratedBertTflite{Name: "bert_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
