package config

import "time"

// GeneratedSchema records execution state for the schema component.
type GeneratedSchema struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSchema() *GeneratedSchema {
	return &GeneratedSchema{Name: "schema", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSchema) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSchema) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
