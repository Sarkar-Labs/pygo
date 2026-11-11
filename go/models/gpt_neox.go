package models

import "time"

// GeneratedGptNeox records execution state for the gpt neox component.
type GeneratedGptNeox struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptNeox() *GeneratedGptNeox {
	return &GeneratedGptNeox{Name: "gpt_neox", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptNeox) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptNeox) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
