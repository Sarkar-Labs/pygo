package config

import "time"

// GeneratedValidator records execution state for the validator component.
type GeneratedValidator struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedValidator() *GeneratedValidator {
	return &GeneratedValidator{Name: "validator", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedValidator) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedValidator) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
