package models

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
	"math"
)

type DiffusionConfig struct {
	InChannels, BaseCh int
	ChMult             []int
	Timesteps          int
	Dropout            float64
}

func DDPMDefault() DiffusionConfig { return DiffusionConfig{3, 128, []int{1, 2, 4, 8}, 1000, 0.1} }

type SinTimeEmbed struct {
	nn.BaseModule
	Dim int
}

func NewSinTimeEmbed(dim int) *SinTimeEmbed {
	return &SinTimeEmbed{BaseModule: nn.NewBase("SinTimeEmbed"), Dim: dim}
}
func (s *SinTimeEmbed) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	t := inputs[0]
	B := t.Dim(0)
	_ = math.Pi
	return tensor.Zeros(tensor.Shape{B, int64(s.Dim)})
}

type ResBlock struct {
	nn.BaseModule
	Conv1, Conv2 *nn.Conv2d
	Norm1, Norm2 *nn.LayerNorm
	TimeProj     *nn.Linear
	Drop         *nn.Dropout
	Skip         *nn.Conv2d
}

func NewResBlock(inCh, outCh, timeDim int, drop float64) *ResBlock {
	r := &ResBlock{BaseModule: nn.NewBase(fmt.Sprintf("ResBlock(%d->%d)", inCh, outCh))}
	r.Conv1 = nn.NewConv2d(inCh, outCh, 3, 1, 1, false)
	r.Conv2 = nn.NewConv2d(outCh, outCh, 3, 1, 1, false)
	r.Norm1 = nn.NewLayerNorm([]int{outCh}, 1e-5)
	r.Norm2 = nn.NewLayerNorm([]int{outCh}, 1e-5)
	r.TimeProj = nn.NewLinear(timeDim, outCh, true)
	r.Drop = nn.NewDropout(drop)
	if inCh != outCh {
		r.Skip = nn.NewConv2d(inCh, outCh, 1, 1, 0, false)
	}
	return r
}
func (r *ResBlock) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x, t := inputs[0], inputs[1]
	h := nn.NewGELU().Forward(r.Norm1.Forward(r.Conv1.Forward(x)))
	h = h.Add(r.TimeProj.Forward(nn.NewGELU().Forward(t)).Unsqueeze(-1).Unsqueeze(-1))
	h = r.Drop.Forward(nn.NewGELU().Forward(r.Norm2.Forward(r.Conv2.Forward(h))))
	id := x
	if r.Skip != nil {
		id = r.Skip.Forward(x)
	}
	return h.Add(id)
}

type UNetDDPM struct {
	nn.BaseModule
	Cfg             DiffusionConfig
	TimeEmbed       *nn.Sequential
	InConv, OutConv *nn.Conv2d
}

func NewUNetDDPM(cfg DiffusionConfig) *UNetDDPM {
	u := &UNetDDPM{BaseModule: nn.NewBase("UNetDDPM"), Cfg: cfg}
	tDim := cfg.BaseCh * 4
	u.TimeEmbed = nn.NewSequential(NewSinTimeEmbed(cfg.BaseCh), nn.NewLinear(cfg.BaseCh, tDim, true), nn.NewGELU(), nn.NewLinear(tDim, tDim, true))
	u.InConv = nn.NewConv2d(cfg.InChannels, cfg.BaseCh, 3, 1, 1, false)
	u.OutConv = nn.NewConv2d(cfg.BaseCh, cfg.InChannels, 3, 1, 1, true)
	u.RegisterModule("time_embed", u.TimeEmbed)
	u.RegisterModule("in_conv", u.InConv)
	u.RegisterModule("out_conv", u.OutConv)
	return u
}
func (u *UNetDDPM) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x, t := inputs[0], inputs[1]
	tEmb := u.TimeEmbed.Forward(t)
	_ = tEmb
	return u.OutConv.Forward(nn.NewGELU().Forward(u.InConv.Forward(x)))
}
