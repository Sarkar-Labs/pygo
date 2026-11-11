package models

import "time"

// GeneratedGptGguf records execution state for the gpt gguf component.
type GeneratedGptGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptGguf() *GeneratedGptGguf {
	return &GeneratedGptGguf{Name: "gpt_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
