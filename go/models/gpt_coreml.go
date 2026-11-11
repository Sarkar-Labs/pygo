package models

import "time"

// GeneratedGptCoreml records execution state for the gpt coreml component.
type GeneratedGptCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptCoreml() *GeneratedGptCoreml {
	return &GeneratedGptCoreml{Name: "gpt_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
