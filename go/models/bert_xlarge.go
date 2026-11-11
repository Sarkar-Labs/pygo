package models

import "time"

// GeneratedBertXlarge records execution state for the bert xlarge component.
type GeneratedBertXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertXlarge() *GeneratedBertXlarge {
	return &GeneratedBertXlarge{Name: "bert_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
