package models

import "time"

// GeneratedBertHuge records execution state for the bert huge component.
type GeneratedBertHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertHuge() *GeneratedBertHuge {
	return &GeneratedBertHuge{Name: "bert_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
