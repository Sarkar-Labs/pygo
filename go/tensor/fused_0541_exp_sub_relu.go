package tensor

import "fmt"

func Fused0541ExpSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.ReLU()
	return r, nil
}
func Fused0541ExpSubReluName() string { return "exp_sub_relu" }
