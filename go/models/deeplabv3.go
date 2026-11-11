package models

import "time"

// GeneratedDeeplabv3 records execution state for the deeplabv3 component.
type GeneratedDeeplabv3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDeeplabv3() *GeneratedDeeplabv3 {
	return &GeneratedDeeplabv3{Name: "deeplabv3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDeeplabv3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDeeplabv3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
