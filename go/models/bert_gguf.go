package models

import "time"

// GeneratedBertGguf records execution state for the bert gguf component.
type GeneratedBertGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertGguf() *GeneratedBertGguf {
	return &GeneratedBertGguf{Name: "bert_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
