package distributed

import (
	"fmt"
	"github.com/pygo/go/tensor"
)

type Backend string

const (
	NCCL Backend = "nccl"
	Gloo Backend = "gloo"
)

type ReduceOp int

const (
	Sum ReduceOp = iota
	Mean
	Max
	Min
)

type ProcessGroup struct {
	rank, worldSize int
	backend         Backend
}

func NewProcessGroup(rank, worldSize int, backend Backend) *ProcessGroup {
	fmt.Printf("ProcessGroup: rank=%d world=%d backend=%s\n", rank, worldSize, backend)
	return &ProcessGroup{rank: rank, worldSize: worldSize, backend: backend}
}
func (pg *ProcessGroup) Rank() int      { return pg.rank }
func (pg *ProcessGroup) WorldSize() int { return pg.worldSize }
func (pg *ProcessGroup) AllReduce(t *tensor.Tensor, op ReduceOp) *tensor.Tensor {
	if op == Mean {
		return t.MulScalar(1.0 / float64(pg.worldSize))
	}
	return t
}
func (pg *ProcessGroup) AllGather(t *tensor.Tensor) []*tensor.Tensor {
	r := make([]*tensor.Tensor, pg.worldSize)
	for i := range r {
		r[i] = t.Clone()
	}
	return r
}
func (pg *ProcessGroup) Broadcast(t *tensor.Tensor, src int) *tensor.Tensor { return t }
func (pg *ProcessGroup) Barrier()                                           {}
func AllReduceMean(t *tensor.Tensor, pg *ProcessGroup) *tensor.Tensor       { return pg.AllReduce(t, Mean) }
