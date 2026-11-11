package deploy

import "time"

// GeneratedLambda records execution state for the lambda component.
type GeneratedLambda struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLambda() *GeneratedLambda {
	return &GeneratedLambda{Name: "lambda", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLambda) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLambda) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
