package data

import "time"

// GeneratedTokenDataset records execution state for the token dataset component.
type GeneratedTokenDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTokenDataset() *GeneratedTokenDataset {
	return &GeneratedTokenDataset{Name: "token_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTokenDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTokenDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
