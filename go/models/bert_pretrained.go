package models

import "time"

// GeneratedBertPretrained records execution state for the bert pretrained component.
type GeneratedBertPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertPretrained() *GeneratedBertPretrained {
	return &GeneratedBertPretrained{Name: "bert_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
