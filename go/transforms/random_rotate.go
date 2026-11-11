package transforms

import "time"

// GeneratedRandomRotate records execution state for the random rotate component.
type GeneratedRandomRotate struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomRotate() *GeneratedRandomRotate {
	return &GeneratedRandomRotate{Name: "random_rotate", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomRotate) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomRotate) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
