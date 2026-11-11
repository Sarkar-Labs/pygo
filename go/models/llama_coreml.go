package models

import "time"

// GeneratedLlamaCoreml records execution state for the llama coreml component.
type GeneratedLlamaCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaCoreml() *GeneratedLlamaCoreml {
	return &GeneratedLlamaCoreml{Name: "llama_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
