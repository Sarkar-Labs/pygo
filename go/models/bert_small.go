package models

import "time"

// GeneratedBertSmall records execution state for the bert small component.
type GeneratedBertSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertSmall() *GeneratedBertSmall {
	return &GeneratedBertSmall{Name: "bert_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
