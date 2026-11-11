package models

import "time"

// GeneratedBertTvm records execution state for the bert tvm component.
type GeneratedBertTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertTvm() *GeneratedBertTvm {
	return &GeneratedBertTvm{Name: "bert_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
