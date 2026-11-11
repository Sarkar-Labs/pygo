package regularizers

import "time"

// GeneratedElasticNet records execution state for the elastic net component.
type GeneratedElasticNet struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedElasticNet() *GeneratedElasticNet {
	return &GeneratedElasticNet{Name: "elastic_net", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedElasticNet) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedElasticNet) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
