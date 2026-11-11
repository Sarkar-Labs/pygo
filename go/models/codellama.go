package models

import "time"

// GeneratedCodellama records execution state for the codellama component.
type GeneratedCodellama struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCodellama() *GeneratedCodellama {
	return &GeneratedCodellama{Name: "codellama", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCodellama) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCodellama) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
