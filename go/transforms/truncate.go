package transforms

import "time"

// GeneratedTruncate records execution state for the truncate component.
type GeneratedTruncate struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTruncate() *GeneratedTruncate {
	return &GeneratedTruncate{Name: "truncate", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTruncate) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTruncate) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
