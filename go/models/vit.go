package models

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
)

type ViTConfig struct {
	ImageSize, PatchSize, InChannels, EmbedDim, Depth, NumHeads, NumClasses int
	MLPRatio, Dropout                                                       float64
}

func ViTTiny() ViTConfig  { return ViTConfig{224, 16, 3, 192, 12, 3, 1000, 4., 0.} }
func ViTSmall() ViTConfig { return ViTConfig{224, 16, 3, 384, 12, 6, 1000, 4., 0.} }
func ViTBase() ViTConfig  { return ViTConfig{224, 16, 3, 768, 12, 12, 1000, 4., 0.} }
func ViTLarge() ViTConfig { return ViTConfig{224, 16, 3, 1024, 24, 16, 1000, 4., 0.} }

type ViTBlock struct {
	nn.BaseModule
	Norm1, Norm2 *nn.LayerNorm
	Attn         *nn.MultiHeadAttention
	MLP          *nn.Sequential
	Drop         *nn.Dropout
}

func NewViTBlock(cfg ViTConfig) *ViTBlock {
	b := &ViTBlock{BaseModule: nn.NewBase("ViTBlock")}
	ff := int(float64(cfg.EmbedDim) * cfg.MLPRatio)
	b.Norm1 = nn.NewLayerNorm([]int{cfg.EmbedDim}, 1e-6)
	b.Attn = nn.NewMultiHeadAttention(cfg.EmbedDim, cfg.NumHeads, cfg.Dropout)
	b.Norm2 = nn.NewLayerNorm([]int{cfg.EmbedDim}, 1e-6)
	b.MLP = nn.NewSequential(nn.NewLinear(cfg.EmbedDim, ff, true), nn.NewGELU(), nn.NewDropout(cfg.Dropout), nn.NewLinear(ff, cfg.EmbedDim, true))
	b.Drop = nn.NewDropout(cfg.Dropout)
	b.RegisterModule("norm1", b.Norm1)
	b.RegisterModule("attn", b.Attn)
	b.RegisterModule("norm2", b.Norm2)
	b.RegisterModule("mlp", b.MLP)
	return b
}
func (b *ViTBlock) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	x = x.Add(b.Drop.Forward(b.Attn.Forward(b.Norm1.Forward(x))))
	return x.Add(b.Drop.Forward(b.MLP.Forward(b.Norm2.Forward(x))))
}

type VisionTransformer struct {
	nn.BaseModule
	Cfg                ViTConfig
	PatchProj          *nn.Linear
	CLSToken, PosEmbed *nn.Parameter
	Drop               *nn.Dropout
	Blocks             []*ViTBlock
	Norm               *nn.LayerNorm
	Head               *nn.Linear
}

func NewViT(cfg ViTConfig) *VisionTransformer {
	v := &VisionTransformer{BaseModule: nn.NewBase(fmt.Sprintf("ViT(%dL,%dH)", cfg.Depth, cfg.NumHeads)), Cfg: cfg}
	pd := cfg.InChannels * cfg.PatchSize * cfg.PatchSize
	nP := (cfg.ImageSize / cfg.PatchSize) * (cfg.ImageSize / cfg.PatchSize)
	v.PatchProj = nn.NewLinear(pd, cfg.EmbedDim, true)
	v.CLSToken = nn.NewParameter(tensor.Zeros(tensor.Shape{1, 1, int64(cfg.EmbedDim)}))
	v.PosEmbed = nn.NewParameter(tensor.Zeros(tensor.Shape{1, int64(nP + 1), int64(cfg.EmbedDim)}))
	v.Drop = nn.NewDropout(cfg.Dropout)
	v.Norm = nn.NewLayerNorm([]int{cfg.EmbedDim}, 1e-6)
	v.Head = nn.NewLinear(cfg.EmbedDim, cfg.NumClasses, true)
	v.Blocks = make([]*ViTBlock, cfg.Depth)
	for i := range v.Blocks {
		v.Blocks[i] = NewViTBlock(cfg)
		v.RegisterModule(fmt.Sprintf("blocks.%d", i), v.Blocks[i])
	}
	v.RegisterParameter("cls_token", v.CLSToken)
	v.RegisterParameter("pos_embed", v.PosEmbed)
	return v
}
func (v *VisionTransformer) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	B := x.Dim(0)
	h := v.PatchProj.Forward(x.Flatten(2, -1).Transpose(1, 2))
	h = tensor.Cat([]*tensor.Tensor{v.CLSToken.Tensor.Expand(B, 1, int64(v.Cfg.EmbedDim)), h}, 1).Add(v.PosEmbed.Tensor)
	h = v.Drop.Forward(h)
	for _, blk := range v.Blocks {
		h = blk.Forward(h)
	}
	return v.Head.Forward(v.Norm.Forward(h).Slice(1, 0, 1, 1).Reshape(tensor.Shape{B, int64(v.Cfg.EmbedDim)}))
}
func ViTTinyModel() *VisionTransformer  { return NewViT(ViTTiny()) }
func ViTSmallModel() *VisionTransformer { return NewViT(ViTSmall()) }
func ViTBaseModel() *VisionTransformer  { return NewViT(ViTBase()) }
func ViTLargeModel() *VisionTransformer { return NewViT(ViTLarge()) }
