package models

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
)

type SAMConfig struct{ ImageSize, PatchSize, EmbedDim, Depth, NumHeads, PromptEmbedDim int }

func SAMBase() SAMConfig  { return SAMConfig{1024, 16, 768, 12, 12, 256} }
func SAMLarge() SAMConfig { return SAMConfig{1024, 16, 1024, 24, 16, 256} }
func SAMHuge() SAMConfig  { return SAMConfig{1024, 16, 1280, 32, 16, 256} }

type SAMImageEncoder struct {
	nn.BaseModule
	PatchEmbed *nn.Linear
	EmbedDim   int
	Blocks     []*ViTBlock
	Neck       *nn.Sequential
}

func NewSAMImageEncoder(cfg SAMConfig) *SAMImageEncoder {
	e := &SAMImageEncoder{BaseModule: nn.NewBase("SAMImageEncoder"), EmbedDim: cfg.EmbedDim}
	e.PatchEmbed = nn.NewLinear(3*cfg.PatchSize*cfg.PatchSize, cfg.EmbedDim, true)
	vitCfg := ViTConfig{ImageSize: cfg.ImageSize, PatchSize: cfg.PatchSize, EmbedDim: cfg.EmbedDim, NumHeads: cfg.NumHeads, Depth: cfg.Depth}
	e.Blocks = make([]*ViTBlock, cfg.Depth)
	for i := range e.Blocks {
		e.Blocks[i] = NewViTBlock(vitCfg)
		e.RegisterModule(fmt.Sprintf("blocks.%d", i), e.Blocks[i])
	}
	e.Neck = nn.NewSequential(nn.NewConv2d(cfg.EmbedDim, cfg.PromptEmbedDim, 1, 1, 0, false))
	e.RegisterModule("patch_embed", e.PatchEmbed)
	e.RegisterModule("neck", e.Neck)
	return e
}
func (e *SAMImageEncoder) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	B := x.Dim(0)
	nP := int64((x.Dim(2) / 16) * (x.Dim(3) / 16))
	h := e.PatchEmbed.Forward(x.Flatten(2, -1).Transpose(1, 2).Reshape(tensor.Shape{B, nP, x.Dim(1) * 16 * 16}))
	for _, blk := range e.Blocks {
		h = blk.Forward(h)
	}
	H := x.Dim(2) / 16
	W := x.Dim(3) / 16
	return e.Neck.Forward(h.Reshape(tensor.Shape{B, H, W, int64(e.EmbedDim)}).Permute([]int{0, 3, 1, 2}))
}

type SAMModel struct {
	nn.BaseModule
	Cfg          SAMConfig
	ImageEncoder *SAMImageEncoder
}

func NewSAM(cfg SAMConfig) *SAMModel {
	m := &SAMModel{BaseModule: nn.NewBase("SAM"), Cfg: cfg}
	m.ImageEncoder = NewSAMImageEncoder(cfg)
	m.RegisterModule("image_encoder", m.ImageEncoder)
	return m
}
func (m *SAMModel) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return m.ImageEncoder.Forward(inputs[0])
}
func SAMBaseModel() *SAMModel  { return NewSAM(SAMBase()) }
func SAMLargeModel() *SAMModel { return NewSAM(SAMLarge()) }
func SAMHugeModel() *SAMModel  { return NewSAM(SAMHuge()) }
