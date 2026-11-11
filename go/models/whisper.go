package models

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
)

type WhisperConfig struct{ NMels, NCtx, NState, NHead, NAudioLayer, NTextLayer, NVocab, NTextCtx int }

func WhisperTiny() WhisperConfig   { return WhisperConfig{80, 1500, 384, 6, 4, 4, 51865, 448} }
func WhisperBase() WhisperConfig   { return WhisperConfig{80, 1500, 512, 8, 6, 6, 51865, 448} }
func WhisperSmall() WhisperConfig  { return WhisperConfig{80, 1500, 768, 12, 12, 12, 51865, 448} }
func WhisperMedium() WhisperConfig { return WhisperConfig{80, 1500, 1024, 16, 24, 24, 51865, 448} }
func WhisperLarge() WhisperConfig  { return WhisperConfig{80, 1500, 1280, 20, 32, 32, 51865, 448} }

type WhisperEncoder struct {
	nn.BaseModule
	Conv1, Conv2 *nn.Conv2d
	Layers       []*nn.TransformerEncoderLayer
	LN           *nn.LayerNorm
}

func NewWhisperEncoder(cfg WhisperConfig) *WhisperEncoder {
	e := &WhisperEncoder{BaseModule: nn.NewBase("WhisperEncoder")}
	e.Conv1 = nn.NewConv2d(cfg.NMels, cfg.NState, 3, 1, 1, false)
	e.Conv2 = nn.NewConv2d(cfg.NState, cfg.NState, 3, 2, 1, false)
	e.LN = nn.NewLayerNorm([]int{cfg.NState}, 1e-5)
	e.Layers = make([]*nn.TransformerEncoderLayer, cfg.NAudioLayer)
	for i := range e.Layers {
		e.Layers[i] = nn.NewTransformerEncoderLayer(cfg.NState, cfg.NHead, cfg.NState*4, 0)
		e.RegisterModule(fmt.Sprintf("layers.%d", i), e.Layers[i])
	}
	return e
}
func (e *WhisperEncoder) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	x = nn.NewGELU().Forward(e.Conv1.Forward(x))
	x = nn.NewGELU().Forward(e.Conv2.Forward(x))
	x = x.Permute([]int{0, 2, 1})
	for _, l := range e.Layers {
		x = l.Forward(x)
	}
	return e.LN.Forward(x)
}

type WhisperModel struct {
	nn.BaseModule
	Cfg     WhisperConfig
	Encoder *WhisperEncoder
}

func NewWhisper(cfg WhisperConfig) *WhisperModel {
	w := &WhisperModel{BaseModule: nn.NewBase("Whisper"), Cfg: cfg}
	w.Encoder = NewWhisperEncoder(cfg)
	w.RegisterModule("encoder", w.Encoder)
	return w
}
func (w *WhisperModel) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return w.Encoder.Forward(inputs[0])
}
func WhisperTinyModel() *WhisperModel  { return NewWhisper(WhisperTiny()) }
func WhisperBaseModel() *WhisperModel  { return NewWhisper(WhisperBase()) }
func WhisperSmallModel() *WhisperModel { return NewWhisper(WhisperSmall()) }
func WhisperLargeModel() *WhisperModel { return NewWhisper(WhisperLarge()) }
