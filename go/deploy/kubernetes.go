package deploy

import "time"

// GeneratedKubernetes records execution state for the kubernetes component.
type GeneratedKubernetes struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedKubernetes() *GeneratedKubernetes {
	return &GeneratedKubernetes{Name: "kubernetes", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedKubernetes) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedKubernetes) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
