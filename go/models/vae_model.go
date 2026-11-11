package models

import "time"

// GeneratedVaeModel records execution state for the vae model component.
type GeneratedVaeModel struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVaeModel() *GeneratedVaeModel {
	return &GeneratedVaeModel{Name: "vae_model", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVaeModel) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVaeModel) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
