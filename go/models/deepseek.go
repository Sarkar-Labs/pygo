package models

import "time"

// GeneratedDeepseek records execution state for the deepseek component.
type GeneratedDeepseek struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDeepseek() *GeneratedDeepseek {
	return &GeneratedDeepseek{Name: "deepseek", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDeepseek) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDeepseek) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
