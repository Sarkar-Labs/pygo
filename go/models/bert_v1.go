package models

import "time"

// GeneratedBertV1 records execution state for the bert v1 component.
type GeneratedBertV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertV1() *GeneratedBertV1 {
	return &GeneratedBertV1{Name: "bert_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
