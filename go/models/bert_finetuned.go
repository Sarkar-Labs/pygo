package models

import "time"

// GeneratedBertFinetuned records execution state for the bert finetuned component.
type GeneratedBertFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBertFinetuned() *GeneratedBertFinetuned {
	return &GeneratedBertFinetuned{Name: "bert_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBertFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBertFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
