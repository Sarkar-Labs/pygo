package tensor

import "fmt"

func Fused0166ReluSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Neg()
	return r, nil
}
func Fused0166ReluSubNegName() string { return "relu_sub_neg" }
