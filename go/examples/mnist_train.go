//go:build examples
// +build examples

package main

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/optim"
	"github.com/pygo/go/tensor"
	"github.com/pygo/go/training"
)

type MNISTNet struct {
	nn.BaseModule
	FC1  *nn.Linear
	BN1  *nn.LayerNorm
	FC2  *nn.Linear
	BN2  *nn.LayerNorm
	FC3  *nn.Linear
	Drop *nn.Dropout
}

func NewMNISTNet() *MNISTNet {
	m := &MNISTNet{BaseModule: nn.NewBase("MNISTNet")}
	m.FC1 = nn.NewLinear(784, 512, true)
	m.BN1 = nn.NewLayerNorm([]int{512}, 1e-5)
	m.FC2 = nn.NewLinear(512, 256, true)
	m.BN2 = nn.NewLayerNorm([]int{256}, 1e-5)
	m.FC3 = nn.NewLinear(256, 10, true)
	m.Drop = nn.NewDropout(0.3)
	m.RegisterModule("fc1", m.FC1)
	m.RegisterModule("fc2", m.FC2)
	m.RegisterModule("fc3", m.FC3)
	return m
}
func (m *MNISTNet) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0].Flatten(1, -1)
	x = m.Drop.Forward(nn.NewGELU().Forward(m.BN1.Forward(m.FC1.Forward(x))))
	x = m.Drop.Forward(nn.NewGELU().Forward(m.BN2.Forward(m.FC2.Forward(x))))
	return m.FC3.Forward(x)
}
func main() {
	fmt.Println("═══ PyGo MNIST Training ═══")
	model := NewMNISTNet()
	total, _ := nn.CountParams(model)
	fmt.Printf("Params: %.2fM\n", float64(total)/1e6)
	opt := optim.NewAdamW(model.Parameters(), 3e-4)
	sched := optim.NewWarmupCosine(opt, 100, 1000)
	cfg := training.DefaultConfig()
	cfg.MaxEpochs = 5
	trainer := training.NewTrainer(model, opt, training.CrossEntropyLoss, cfg, training.WithScheduler(sched))
	out := model.Forward(tensor.Randn(tensor.Shape{32, 1, 28, 28}))
	fmt.Printf("Output: %v\n", out.Shape())
	_ = trainer
}
