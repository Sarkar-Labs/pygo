package autograd

import "time"

// GeneratedChainRule records execution state for the chain rule component.
type GeneratedChainRule struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedChainRule() *GeneratedChainRule {
	return &GeneratedChainRule{Name: "chain_rule", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedChainRule) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedChainRule) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
