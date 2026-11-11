package models

import "time"

// GeneratedGanModel records execution state for the gan model component.
type GeneratedGanModel struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGanModel() *GeneratedGanModel {
	return &GeneratedGanModel{Name: "gan_model", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGanModel) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGanModel) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
