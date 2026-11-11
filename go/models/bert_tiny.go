package models

import "time"

// GeneratedBertTiny records execution state for the bert tiny component.
type GeneratedBertTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertTiny() *GeneratedBertTiny {
	return &GeneratedBertTiny{Name: "bert_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
