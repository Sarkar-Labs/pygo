package tensor

import "fmt"

func Fused0460SqrtAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Tanh()
	return r, nil
}
func Fused0460SqrtAddTanhName() string { return "sqrt_add_tanh" }
