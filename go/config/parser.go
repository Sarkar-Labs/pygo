package config

import "time"

// GeneratedParser records execution state for the parser component.
type GeneratedParser struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedParser() *GeneratedParser {
	return &GeneratedParser{Name: "parser", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedParser) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedParser) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
