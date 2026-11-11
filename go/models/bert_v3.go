package models

import "time"

// GeneratedBertV3 records execution state for the bert v3 component.
type GeneratedBertV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertV3() *GeneratedBertV3 {
	return &GeneratedBertV3{Name: "bert_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
