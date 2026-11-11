package models

import "time"

// GeneratedBertSafetensors records execution state for the bert safetensors component.
type GeneratedBertSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertSafetensors() *GeneratedBertSafetensors {
	return &GeneratedBertSafetensors{Name: "bert_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
