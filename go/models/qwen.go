package models

import "time"

// GeneratedQwen records execution state for the qwen component.
type GeneratedQwen struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedQwen() *GeneratedQwen { return &GeneratedQwen{Name: "qwen", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedQwen) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedQwen) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
