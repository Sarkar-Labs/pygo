package data

import "time"

// GeneratedCompose records execution state for the compose component.
type GeneratedCompose struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCompose() *GeneratedCompose {
	return &GeneratedCompose{Name: "compose", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCompose) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCompose) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
