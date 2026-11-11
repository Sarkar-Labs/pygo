package training

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
)

type PipelineParallel struct {
	stages   [][]nn.Module
	devices  []tensor.Device
	numMicro int
}

func NewPipelineParallel(layers []nn.Module, devices []tensor.Device, numMicro int) *PipelineParallel {
	pp := &PipelineParallel{devices: devices, numMicro: numMicro}
	per := (len(layers) + len(devices) - 1) / len(devices)
	for i := 0; i < len(layers); i += per {
		end := i + per
		if end > len(layers) {
			end = len(layers)
		}
		pp.stages = append(pp.stages, layers[i:end])
	}
	fmt.Printf("Pipeline: %d stages %d micro\n", len(pp.stages), numMicro)
	return pp
}
func (pp *PipelineParallel) Forward(x *tensor.Tensor) *tensor.Tensor {
	B := x.Dim(0)
	msz := B / int64(pp.numMicro)
	outs := make([]*tensor.Tensor, pp.numMicro)
	for m := 0; m < pp.numMicro; m++ {
		s := int64(m) * msz
		h := x.Slice(0, s, s+msz, 1)
		for si, stage := range pp.stages {
			h = h.To(pp.devices[si])
			for _, l := range stage {
				h = l.Forward(h)
			}
		}
		outs[m] = h
	}
	return tensor.Cat(outs, 0)
}
