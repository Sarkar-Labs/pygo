package tensor

import "fmt"

func Fused0111ReluSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Sqrt()
	return r, nil
}
func Fused0111ReluSubSqrtName() string { return "relu_sub_sqrt" }
