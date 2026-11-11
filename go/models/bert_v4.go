package models

import "time"

// GeneratedBertV4 records execution state for the bert v4 component.
type GeneratedBertV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertV4() *GeneratedBertV4 {
	return &GeneratedBertV4{Name: "bert_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
