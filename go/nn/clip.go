package nn

import (
	"fmt"
	"github.com/pygo/go/tensor"
	"math"
)

type CLIPConfig struct {
	VocabSize, MaxTextLen, TextDim, TextHeads, TextLayers, ImageSize, PatchSize, ImageDim, ImageHeads, ImageLayers, ProjectionDim int
	Temperature                                                                                                                   float64
}

func CLIPBase() CLIPConfig { return CLIPConfig{49408, 77, 512, 8, 12, 224, 32, 768, 12, 12, 512, 0.07} }

type CLIPModel struct {
	BaseModule
	Cfg        CLIPConfig
	LogitScale *Parameter
}

func NewCLIP(cfg CLIPConfig) *CLIPModel {
	m := &CLIPModel{BaseModule: NewBase(fmt.Sprintf("CLIP(proj=%d)", cfg.ProjectionDim)), Cfg: cfg}
	m.LogitScale = NewParameter(tensor.FromSlice([]float32{float32(math.Log(1.0 / cfg.Temperature))}, tensor.Shape{1}))
	m.RegisterParameter("logit_scale", m.LogitScale)
	return m
}
func (m *CLIPModel) EncodeText(ids *tensor.Tensor) *tensor.Tensor {
	feat := tensor.Randn(tensor.Shape{ids.Dim(0), int64(m.Cfg.ProjectionDim)})
	return feat.Div(feat.Norm(2, -1, true).AddScalar(1e-8))
}
func (m *CLIPModel) EncodeImage(imgs *tensor.Tensor) *tensor.Tensor {
	feat := tensor.Randn(tensor.Shape{imgs.Dim(0), int64(m.Cfg.ProjectionDim)})
	return feat.Div(feat.Norm(2, -1, true).AddScalar(1e-8))
}
func (m *CLIPModel) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	img := m.EncodeImage(inputs[0])
	txt := m.EncodeText(inputs[1])
	return img.MatMul(txt.Transpose(-2, -1)).MulScalar(m.LogitScale.Tensor.Exp().Item())
}
func CLIPBaseModel() *CLIPModel { return NewCLIP(CLIPBase()) }
