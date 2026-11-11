package models

import "time"

// GeneratedBertV2 records execution state for the bert v2 component.
type GeneratedBertV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertV2() *GeneratedBertV2 {
	return &GeneratedBertV2{Name: "bert_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
