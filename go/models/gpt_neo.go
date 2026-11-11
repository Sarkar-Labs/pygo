package models

import "time"

// GeneratedGptNeo records execution state for the gpt neo component.
type GeneratedGptNeo struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptNeo() *GeneratedGptNeo {
	return &GeneratedGptNeo{Name: "gpt_neo", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptNeo) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptNeo) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
