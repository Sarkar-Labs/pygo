package models

import "time"

// GeneratedBertBase records execution state for the bert base component.
type GeneratedBertBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertBase() *GeneratedBertBase {
	return &GeneratedBertBase{Name: "bert_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
