package deploy

import "time"

// GeneratedGke records execution state for the gke component.
type GeneratedGke struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGke() *GeneratedGke { return &GeneratedGke{Name: "gke", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGke) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGke) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
