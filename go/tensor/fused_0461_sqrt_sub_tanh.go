package tensor

import "fmt"

func Fused0461SqrtSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Tanh()
	return r, nil
}
func Fused0461SqrtSubTanhName() string { return "sqrt_sub_tanh" }
