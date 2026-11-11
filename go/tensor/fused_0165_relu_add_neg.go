package tensor

import "fmt"

func Fused0165ReluAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Neg()
	return r, nil
}
func Fused0165ReluAddNegName() string { return "relu_add_neg" }
