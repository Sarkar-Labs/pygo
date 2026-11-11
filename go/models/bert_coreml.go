package models

import "time"

// GeneratedBertCoreml records execution state for the bert coreml component.
type GeneratedBertCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertCoreml() *GeneratedBertCoreml {
	return &GeneratedBertCoreml{Name: "bert_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
