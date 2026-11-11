//go:build examples
// +build examples

package main

import (
	"fmt"
	"github.com/pygo/go/models"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/optim"
	"github.com/pygo/go/tensor"
)

type BERTClassifier struct {
	nn.BaseModule
	BERT       *models.BERTModel
	Drop       *nn.Dropout
	Classifier *nn.Linear
}

func NewBERTClassifier(numLabels int) *BERTClassifier {
	cfg := models.BERTBase()
	c := &BERTClassifier{BaseModule: nn.NewBase(fmt.Sprintf("BERTClassifier(%d)", numLabels))}
	c.BERT = models.NewBERT(cfg)
	c.Drop = nn.NewDropout(0.1)
	c.Classifier = nn.NewLinear(cfg.HiddenSize, numLabels, true)
	c.RegisterModule("bert", c.BERT)
	c.RegisterModule("classifier", c.Classifier)
	return c
}
func (c *BERTClassifier) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	h := c.BERT.Forward(inputs[0])
	cls := h.Slice(1, 0, 1, 1).Squeeze(1)
	return c.Classifier.Forward(c.Drop.Forward(cls))
}
func main() {
	fmt.Println("═══ PyGo BERT Fine-tuning ═══")
	model := NewBERTClassifier(2)
	total, trainable := nn.CountParams(model)
	fmt.Printf("Params: %.2fM total | %.2fM trainable\n", float64(total)/1e6, float64(trainable)/1e6)
	opt := optim.NewAdamW(model.Parameters(), 2e-5, optim.WithWD(0.01))
	out := model.Forward(tensor.Zeros(tensor.Shape{4, 128}))
	fmt.Printf("Logits: %v\n", out.Shape())
	_ = opt
	fmt.Println("Ready!")
}
