package models

import "time"

// GeneratedBertDistilled records execution state for the bert distilled component.
type GeneratedBertDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertDistilled() *GeneratedBertDistilled {
	return &GeneratedBertDistilled{Name: "bert_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
