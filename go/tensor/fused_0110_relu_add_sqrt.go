package tensor

import "fmt"

func Fused0110ReluAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Sqrt()
	return r, nil
}
func Fused0110ReluAddSqrtName() string { return "relu_add_sqrt" }
