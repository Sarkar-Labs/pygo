package inference

import "time"

// GeneratedPrefixAllowed records execution state for the prefix allowed component.
type GeneratedPrefixAllowed struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPrefixAllowed() *GeneratedPrefixAllowed {
	return &GeneratedPrefixAllowed{Name: "prefix_allowed", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPrefixAllowed) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPrefixAllowed) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
