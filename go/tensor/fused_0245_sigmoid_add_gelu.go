package tensor

import "fmt"

func Fused0245SigmoidAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.GELU()
	return r, nil
}
func Fused0245SigmoidAddGeluName() string { return "sigmoid_add_gelu" }
