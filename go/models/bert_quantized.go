package models

import "time"

// GeneratedBertQuantized records execution state for the bert quantized component.
type GeneratedBertQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertQuantized() *GeneratedBertQuantized {
	return &GeneratedBertQuantized{Name: "bert_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
