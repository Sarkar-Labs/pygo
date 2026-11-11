package transforms

import "time"

// GeneratedAddSpecialTokens records execution state for the add special tokens component.
type GeneratedAddSpecialTokens struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAddSpecialTokens() *GeneratedAddSpecialTokens {
	return &GeneratedAddSpecialTokens{Name: "add_special_tokens", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAddSpecialTokens) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAddSpecialTokens) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
